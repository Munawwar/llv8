// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LOWCHUNK_H_
#define V8_LOWCHUNK_H_

//#include "compiler.h"
#include "hydrogen.h"
//#include "zone.h"


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

class LowChunkBuilderBase BASE_EMBEDDED {
  public:
    virtual ~LowChunkBuilderBase() {}
    explicit LowChunkBuilderBase(CompilationInfo* info, HGraph* graph)
        : chunk_(nullptr),
          info_(info),
          graph_(graph),
          zone_(graph->zone()) {}

  protected:
    virtual LowChunk* chunk() const { return chunk_; }
    CompilationInfo* info() const { return info_; }
    HGraph* graph() const { return graph_; }
    Isolate* isolate() const { return graph_->isolate(); }
    Heap* heap() const { return isolate()->heap(); }
    Zone* zone() const { return zone_; }

    LowChunk* chunk_;
    CompilationInfo* info_;
    HGraph* const graph_;

  private:
   Zone* zone_;
};

} }  // namespace v8::internal

#endif  // V8_LOWCHUNK_H_
