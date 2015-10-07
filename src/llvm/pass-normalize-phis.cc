// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "pass-normalize-phis.h"

#include "src/base/macros.h"
#include <set>
//#include "src/globals.h"
//#include "src/list-inl.h"

namespace v8 {
namespace internal {

// FunctionPasses may overload three virtual methods to do their work.
// All of these methods should return true if they modified the program,
// or false if they didn’t.
class NormalizePhisPass : public llvm::FunctionPass {
 public:
  NormalizePhisPass();
  bool runOnFunction(llvm::Function& function) override;
  void getAnalysisUsage(llvm::AnalysisUsage& analysis_usage) const override;

//  bool doInitialization(Module& module) override { return false; };
  static char ID;
};

char NormalizePhisPass::ID = 0;

NormalizePhisPass::NormalizePhisPass() : FunctionPass(ID) {}

bool NormalizePhisPass::runOnFunction(llvm::Function& function) {
  auto debug = false;
#ifdef DEBUG
  debug = true;
#endif
  auto changed = false;
  llvm::DominatorTree& dom_tree = getAnalysis<llvm::DominatorTreeWrapperPass>()
      .getDomTree();
  if (debug) dom_tree.verifyDomTree();

  // for each BB in the function
  for (auto bb = function.begin(); bb != function.end(); ++bb) {
    if (debug) std::cerr << "Grabbed a new BB\n";
    llvm::PHINode* phi;
    // for all phi nodes in the block
    for (auto it = bb->begin(); (phi = llvm::dyn_cast<llvm::PHINode>(it));
        ++it) {
      if (debug) std::cerr << "Grabbed a new Phi\n";
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
        if (debug)
          std::cerr << "SIZE BEFORE " << wrongs.size() - rights.size() << "\n";

        for (auto wrong_pair : wrongs) {
          if (rights.count(wrong_pair.first)) continue;
          auto wrong_node = dom_tree.getNode(wrong_pair.first);
          llvm::BasicBlock* unique_dominated = nullptr;
          int dominated = 0;
          bool no_choice = (preds.size() == 1);
          if (no_choice) {
            unique_dominated = *preds.begin(); // here it might not be dominated
          } else {
            for (auto b : preds) {
              if (dom_tree.dominates(wrong_node, dom_tree.getNode(b))) {
                dominated++;
                unique_dominated = b;
              }
              if (dominated > 1) break;
            }
          }
          if (dominated == 1 || no_choice) {
            phi->setIncomingBlock(wrong_pair.second, unique_dominated);
            rights.insert(wrong_pair.first); // effectively remove from wrongs
            preds.erase(unique_dominated); // remove from preds
            changed = true;
          }
        }
        if (debug)
          std::cerr << "SIZE AFTER " << wrongs.size() - rights.size() << "\n";
      } // while there are wrong blocks left
    } // for all phi nodes in the block
  } // for each BB in the function
  return changed;
}

void NormalizePhisPass::getAnalysisUsage(
    llvm::AnalysisUsage& analysis_usage) const {
  analysis_usage.addRequired<llvm::DominatorTreeWrapperPass>();
  analysis_usage.setPreservesAll();
}

llvm::FunctionPass* createNormalizePhisPass() {
  llvm::initializeDominatorTreeWrapperPassPass(
      *llvm::PassRegistry::getPassRegistry());
  return new NormalizePhisPass();
}

} }  // namespace v8::internal
