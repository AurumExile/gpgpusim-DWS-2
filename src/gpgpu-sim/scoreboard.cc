// --- IN scoreboard.cc ---

#include "scoreboard.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader.h"
#include "shader_trace.h"

// Constructor
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu)
    : longopregs() {
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);

  m_gpu = gpu;
}

// Print scoreboard contents (Modified for map)
void Scoreboard::printContents() const {
  printf("scoreboard contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    printf("  wid = %2d: ", i);
    std::map<unsigned, active_mask_t>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("%u(%s) ", it->first, it->second.to_string().c_str());
    printf("\n");
  }
}

// DWS: Reserve Register per-thread
void Scoreboard::reserveRegister(unsigned wid, unsigned regnum,
                                 const active_mask_t &mask) {
  // If the register isn't locked at all yet, initialize it
  if (reg_table[wid].find(regnum) == reg_table[wid].end()) {
    reg_table[wid][regnum] = mask;
  } else {
    // If it is locked, ensure the exact same threads aren't trying to lock it
    // again
    if ((reg_table[wid][regnum] & mask).any()) {
      printf(
          "Error: trying to reserve an already reserved register by same "
          "threads (sid=%d, wid=%d, regnum=%d).\n",
          m_sid, wid, regnum);
      abort();
    }
    // Bitwise OR to add these threads to the lock
    reg_table[wid][regnum] |= mask;
  }
  SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d, mask: %s\n",
                 wid, regnum, mask.to_string().c_str());
}

// DWS: Unmark register as write-pending per-thread
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum,
                                 const active_mask_t &mask) {
  if (reg_table[wid].find(regnum) == reg_table[wid].end()) return;

  // Bitwise AND NOT to clear only the threads that just finished
  reg_table[wid][regnum] &= ~(mask);

  SHADER_DPRINTF(SCOREBOARD,
                 "Release register - warp:%d, reg: %d, mask freed: %s\n", wid,
                 regnum, mask.to_string().c_str());

  // If no threads are locking this register anymore, erase it from the map
  // completely
  if (reg_table[wid][regnum].none()) {
    reg_table[wid].erase(regnum);
  }
}

// DWS: Check long op status
const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

// DWS: Bulk reserve from instruction
void Scoreboard::reserveRegisters(const class warp_inst_t *inst) {
  const active_mask_t &mask = inst->get_active_mask();

  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegister(inst->warp_id(), inst->out[r], mask);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d, mask: %d\n",
                     inst->warp_id(), inst->out[r], mask);
    }
  }

  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        // Initialize or bitwise OR the mask into the longop map
        if (longopregs[inst->warp_id()].find(inst->out[r]) ==
            longopregs[inst->warp_id()].end()) {
          longopregs[inst->warp_id()][inst->out[r]] = mask;
        } else {
          longopregs[inst->warp_id()][inst->out[r]] |= mask;
        }
      }
    }
  }
}

// DWS: Bulk release from instruction
void Scoreboard::releaseRegisters(const class warp_inst_t *inst) {
  const active_mask_t &mask = inst->get_active_mask();

  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      releaseRegister(inst->warp_id(), inst->out[r], mask);

      // Clean up long op tracker
      if (longopregs[inst->warp_id()].find(inst->out[r]) !=
          longopregs[inst->warp_id()].end()) {
        longopregs[inst->warp_id()][inst->out[r]] &= ~(mask);
        if (longopregs[inst->warp_id()][inst->out[r]].none()) {
          longopregs[inst->warp_id()].erase(inst->out[r]);
        }
      }
    }
  }
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 * scoreboard DWS: Collision only occurs if the active threads of the issuing
 * instruction overlap with the threads that currently hold the lock on that
 * register.
 **/
bool Scoreboard::checkCollision(unsigned wid, const class inst_t *inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Grab the specific mask of the split trying to issue
  const class warp_inst_t *warp_inst =
      static_cast<const class warp_inst_t *>(inst);
  const active_mask_t &issuing_mask = warp_inst->get_active_mask();

  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++) {
    // If the register exists in the table...
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      // DWS: AND the active mask of the lock with the active mask of the
      // instruction
      if ((reg_table[wid].at(*it2) & issuing_mask).any()) {
        return true;  // Collision! Threads overlap.
      }
    }
  }
  return false;  // Safe to issue for these specific threads
}

bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();
}