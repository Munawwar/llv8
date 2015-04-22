// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "llvm-passes.h"

#include "src/base/macros.h"
#include <set>
//#include "src/globals.h"
//#include "src/list-inl.h"

namespace v8 {
namespace internal {

char NormalizePhisPass::ID = 0;

static void* initializeNormalizePhisPassPassOnce(llvm::PassRegistry &Registry) {
  initializeDominatorTreeWrapperPassPass(Registry);

  llvm::PassInfo *PI = new llvm::PassInfo("Normalize phis", "normalize-phis", &NormalizePhisPass::ID,
      llvm::PassInfo::NormalCtor_t(llvm::callDefaultCtor<NormalizePhisPass>), false, false);
  Registry.registerPass(*PI, true);
  return PI;
}

void initializeNormalizePhisPassPass(llvm::PassRegistry &Registry) {
  static volatile llvm::sys::cas_flag initialized = 0;
  llvm::sys::cas_flag old_val = llvm::sys::CompareAndSwap(&initialized, 1, 0);
  if (old_val == 0) {
    initializeNormalizePhisPassPassOnce(Registry);
    llvm::sys::MemoryFence();
    initialized = 2;
  } else {
    llvm::sys::cas_flag tmp = initialized;
    llvm::sys::MemoryFence();
    while (tmp != 2) {
      tmp = initialized;
      llvm::sys::MemoryFence();
    }
  }
}

// TODO(llvm): the above 2 functions are handwritten expansion of llvm
// INITIALIZE_PASS_DEPENDENCY and INITIALIZE_PASS_END macros.
// (See for example llvm/lib/Transforms/IPO/LoopExtractor.cpp)
// The commented line below should to the same work but for some reason it doesn't.

//static llvm::RegisterPass<NormalizePhisPass> register_normalize_phis("normalizePhis", "Normalize phis", true, true);

bool NormalizePhisPass::runOnFunction(llvm::Function& function) {
  auto changed = false;
  llvm::DominatorTree& dom_tree = getAnalysis<llvm::DominatorTreeWrapperPass>()
      .getDomTree();
#ifdef DEBUG
  dom_tree.verifyDomTree();
#endif

  // for each BB in the function
  for (auto bb = function.begin(); bb != function.end(); ++bb) {
    llvm::PHINode* phi;
    // for all phi nodes in the block
    for (auto it = bb->begin(); (phi = llvm::dyn_cast<llvm::PHINode>(it));
        ++it) {

      // FIXME(llvm): v8 doesn't like STL much
      std::set<llvm::BasicBlock*> preds(llvm::pred_begin(bb),
                                        llvm::pred_end(bb));
      std::set<llvm::BasicBlock*> rights;
      std::map<llvm::BasicBlock*, unsigned> wrongs;

      // for each phi input
      for (auto i = 0; i < phi->getNumIncomingValues(); ++i) {
        llvm::BasicBlock* incoming = phi->getIncomingBlock(i);
        if (preds.count(incoming))
          preds.erase(incoming);
        else
          wrongs[incoming] = i;
      }

      while (wrongs.size() > rights.size()) {
        // FIXME(llvm):
        // 1) if a loop is gonna run endlessly, fail
        // 2) case if there is no block with dominated == 1
        std::cerr << "SIZE BEFORE " << wrongs.size() - rights.size() << std::endl;
        for (auto wrong_pair : wrongs) {
          if (rights.count(wrong_pair.first)) continue;
          auto wrong_node = dom_tree.getNode(wrong_pair.first);
          llvm::BasicBlock* unique_dominated = nullptr;
          int dominated = 0;
          for (auto b : preds) {
            if (dom_tree.dominates(wrong_node, dom_tree.getNode(b))) {
              dominated++;
              unique_dominated = b;
            }
            if (dominated > 1) break;
          }
          if (dominated == 1) {
            phi->setIncomingBlock(wrong_pair.second, unique_dominated);
            rights.insert(wrong_pair.first); // effectively remove from wrongs
            preds.erase(unique_dominated); // remove from preds
            changed = true;
          }
        }
        std::cerr << "SIZE AFTER " << wrongs.size() - rights.size() << std::endl;
      } // while there are wrong blocks left
    } // for all phi nodes in the block
  } // for each BB in the function
  return changed;
}

void NormalizePhisPass::getAnalysisUsage(llvm::AnalysisUsage& analysis_usage) const {
  analysis_usage.setPreservesAll();
  analysis_usage.addRequired<llvm::DominatorTreeWrapperPass>();
}

} }  // namespace v8::internal
