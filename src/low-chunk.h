// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LOWCHUNK_H_
#define V8_LOWCHUNK_H_

#include "zone.h"
#include "hydrogen.h"
#include "compiler.h"


namespace v8 {
namespace internal {

class LowChunk : public ZoneObject {
  public:
    virtual ~LowChunk() {}
    //virtual LowChunk* NewChunk(HGraph *graph) = 0;
    virtual Handle<Code> Codegen() = 0;
    Zone* zone() const { return info_->zone(); }

    CompilationInfo* info() const { return info_; }
    HGraph* graph() const { return graph_; }
    Isolate* isolate() const { return graph_->isolate(); }

  protected:
    LowChunk(CompilationInfo* info, HGraph* graph);

  private:
    CompilationInfo* info_;
    HGraph* const graph_;
};

}  // namespace v8
}  // namespace internal

#endif  // V8_LOWCHUNK_H_
