// |jit-test| test-also-noasmjs
/* -*- Mode: javascript; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 ; js-indent-level : 2 ; js-curly-indent-offset: 0 -*- */
/* vim: set ts=2 et sw=2 tw=80: */

// Mandelbrot using SIMD
// Author: Peter Jensen, Intel Corporation

// In polyfill mode, uncomment these two lines + comment "use asm"
SIMD = {};


/*
  Copyright (C) 2013

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

if (typeof SIMD === "undefined") {
  // SIMD module. We don't use the var keyword here, so that we put the
  // SIMD object in the global scope even if this polyfill code is included
  // within some other scope. The theory is that we're anticipating a
  // future where SIMD is predefined in the global scope.
  SIMD = {};
}

// private stuff.
var _SIMD_PRIVATE = {};

_SIMD_PRIVATE._f32x4 = new Float32Array(4);
_SIMD_PRIVATE._f64x2 = new Float64Array(_SIMD_PRIVATE._f32x4.buffer);
_SIMD_PRIVATE._i32x4 = new Int32Array(_SIMD_PRIVATE._f32x4.buffer);
_SIMD_PRIVATE._i16x8 = new Int16Array(_SIMD_PRIVATE._f32x4.buffer);
_SIMD_PRIVATE._i8x16 = new Int8Array(_SIMD_PRIVATE._f32x4.buffer);

_SIMD_PRIVATE._f32x8 = new Float32Array(8);
_SIMD_PRIVATE._f64x4 = new Float64Array(4);
_SIMD_PRIVATE._i32x8 = new Int32Array(8);
_SIMD_PRIVATE._i16x16 = new Int16Array(16);
_SIMD_PRIVATE._i8x32 = new Int8Array(32);

if (typeof Math.fround !== 'undefined') {
  _SIMD_PRIVATE.truncatef32 = Math.fround;
} else {
  _SIMD_PRIVATE._f32 = new Float32Array(1);

  _SIMD_PRIVATE.truncatef32 = function(x) {
    _SIMD_PRIVATE._f32[0] = x;
    return _SIMD_PRIVATE._f32[0];
  }
}

_SIMD_PRIVATE.isNumber = function(o) {
  return typeof o === "number" || (typeof o === "object" && o.constructor === Number);
}

_SIMD_PRIVATE.isTypedArray = function(o) {
  return (o instanceof Int8Array) ||
         (o instanceof Uint8Array) ||
         (o instanceof Uint8ClampedArray) ||
         (o instanceof Int16Array) ||
         (o instanceof Uint16Array) ||
         (o instanceof Int32Array) ||
         (o instanceof Uint32Array) ||
         (o instanceof Float32Array) ||
         (o instanceof Float64Array) ||
         (o instanceof Int32x4Array) ||
         (o instanceof Float32x4Array);
}

_SIMD_PRIVATE.isArrayBuffer = function(o) {
  return (o instanceof ArrayBuffer);
}

_SIMD_PRIVATE.minNum = function(x, y) {
  return x != x ? y :
         y != y ? x :
         Math.min(x, y);
}

_SIMD_PRIVATE.maxNum = function(x, y) {
  return x != x ? y :
         y != y ? x :
         Math.max(x, y);
}

_SIMD_PRIVATE.tobool = function(x) {
  return x < 0;
}

_SIMD_PRIVATE.frombool = function(x) {
  return !x - 1;
}

// Save/Restore utilities for implementing bitwise conversions.

_SIMD_PRIVATE.saveFloat64x2 = function(x) {
  x = SIMD.float64x2.check(x);
  _SIMD_PRIVATE._f64x2[0] = x.x;
  _SIMD_PRIVATE._f64x2[1] = x.y;
}

_SIMD_PRIVATE.saveFloat32x4 = function(x) {
  x = SIMD.float32x4.check(x);
  _SIMD_PRIVATE._f32x4[0] = x.x;
  _SIMD_PRIVATE._f32x4[1] = x.y;
  _SIMD_PRIVATE._f32x4[2] = x.z;
  _SIMD_PRIVATE._f32x4[3] = x.w;
}

_SIMD_PRIVATE.saveInt32x4 = function(x) {
  x = SIMD.int32x4.check(x);
  _SIMD_PRIVATE._i32x4[0] = x.x;
  _SIMD_PRIVATE._i32x4[1] = x.y;
  _SIMD_PRIVATE._i32x4[2] = x.z;
  _SIMD_PRIVATE._i32x4[3] = x.w;
}

_SIMD_PRIVATE.saveInt16x8 = function(x) {
  x = SIMD.int16x8.check(x);
  _SIMD_PRIVATE._i16x8[0] = x.s0;
  _SIMD_PRIVATE._i16x8[1] = x.s1;
  _SIMD_PRIVATE._i16x8[2] = x.s2;
  _SIMD_PRIVATE._i16x8[3] = x.s3;
  _SIMD_PRIVATE._i16x8[4] = x.s4;
  _SIMD_PRIVATE._i16x8[5] = x.s5;
  _SIMD_PRIVATE._i16x8[6] = x.s6;
  _SIMD_PRIVATE._i16x8[7] = x.s7;
}

_SIMD_PRIVATE.saveInt8x16 = function(x) {
  x = SIMD.int8x16.check(x);
  _SIMD_PRIVATE._i8x16[0] = x.s0;
  _SIMD_PRIVATE._i8x16[1] = x.s1;
  _SIMD_PRIVATE._i8x16[2] = x.s2;
  _SIMD_PRIVATE._i8x16[3] = x.s3;
  _SIMD_PRIVATE._i8x16[4] = x.s4;
  _SIMD_PRIVATE._i8x16[5] = x.s5;
  _SIMD_PRIVATE._i8x16[6] = x.s6;
  _SIMD_PRIVATE._i8x16[7] = x.s7;
  _SIMD_PRIVATE._i8x16[8] = x.s8;
  _SIMD_PRIVATE._i8x16[9] = x.s9;
  _SIMD_PRIVATE._i8x16[10] = x.s10;
  _SIMD_PRIVATE._i8x16[11] = x.s11;
  _SIMD_PRIVATE._i8x16[12] = x.s12;
  _SIMD_PRIVATE._i8x16[13] = x.s13;
  _SIMD_PRIVATE._i8x16[14] = x.s14;
  _SIMD_PRIVATE._i8x16[15] = x.s15;
}

_SIMD_PRIVATE.restoreFloat64x2 = function() {
  var alias = _SIMD_PRIVATE._f64x2;
  return SIMD.float64x2(alias[0], alias[1]);
}

_SIMD_PRIVATE.restoreFloat32x4 = function() {
  var alias = _SIMD_PRIVATE._f32x4;
  return SIMD.float32x4(alias[0], alias[1], alias[2], alias[3]);
}

_SIMD_PRIVATE.restoreInt32x4 = function() {
  var alias = _SIMD_PRIVATE._i32x4;
  return SIMD.int32x4(alias[0], alias[1], alias[2], alias[3]);
}

_SIMD_PRIVATE.restoreInt16x8 = function() {
  var alias = _SIMD_PRIVATE._i16x8;
  return SIMD.int16x8(alias[0], alias[1], alias[2], alias[3],
                      alias[4], alias[5], alias[6], alias[7]);
}

_SIMD_PRIVATE.restoreInt8x16 = function() {
  var alias = _SIMD_PRIVATE._i8x16;
  return SIMD.int8x16(alias[0], alias[1], alias[2], alias[3],
                      alias[4], alias[5], alias[6], alias[7],
                      alias[8], alias[9], alias[10], alias[11],
                      alias[12], alias[13], alias[14], alias[15]);
}

if (typeof SIMD.float32x4 === "undefined") {
  /**
    * Construct a new instance of float32x4 number.
    * @param {double} value used for x lane.
    * @param {double} value used for y lane.
    * @param {double} value used for z lane.
    * @param {double} value used for w lane.
    * @constructor
    */
  SIMD.float32x4 = function(x, y, z, w) {
    if (!(this instanceof SIMD.float32x4)) {
      return new SIMD.float32x4(x, y, z, w);
    }

    this.x_ = _SIMD_PRIVATE.truncatef32(x);
    this.y_ = _SIMD_PRIVATE.truncatef32(y);
    this.z_ = _SIMD_PRIVATE.truncatef32(z);
    this.w_ = _SIMD_PRIVATE.truncatef32(w);
  }

  Object.defineProperty(SIMD.float32x4.prototype, 'x', {
    get: function() { return this.x_; }
  });

  Object.defineProperty(SIMD.float32x4.prototype, 'y', {
    get: function() { return this.y_; }
  });

  Object.defineProperty(SIMD.float32x4.prototype, 'z', {
    get: function() { return this.z_; }
  });

  Object.defineProperty(SIMD.float32x4.prototype, 'w', {
    get: function() { return this.w_; }
  });

  /**
    * Extract the sign bit from each lane return them in the first 4 bits.
    */
  Object.defineProperty(SIMD.float32x4.prototype, 'signMask', {
    get: function() {
      var mx = (this.x < 0.0 || 1/this.x === -Infinity);
      var my = (this.y < 0.0 || 1/this.y === -Infinity);
      var mz = (this.z < 0.0 || 1/this.z === -Infinity);
      var mw = (this.w < 0.0 || 1/this.w === -Infinity);
      return mx | my << 1 | mz << 2 | mw << 3;
    }
  });
}

if (typeof SIMD.float32x4.check === "undefined") {
  /**
    * Check whether the argument is a float32x4.
    * @param {float32x4} v An instance of float32x4.
    * @return {float32x4} The float32x4 instance.
    */
  SIMD.float32x4.check = function(v) {
    if (!(v instanceof SIMD.float32x4)) {
      throw new TypeError("argument is not a float32x4.");
    }
    return v;
  }
}

if (typeof SIMD.float32x4.splat === "undefined") {
  /**
    * Construct a new instance of float32x4 number with the same value
    * in all lanes.
    * @param {double} value used for all lanes.
    * @constructor
    */
  SIMD.float32x4.splat = function(s) {
    return SIMD.float32x4(s, s, s, s);
  }
}

if (typeof SIMD.float32x4.fromFloat64x2 === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @return {float32x4} A float32x4 with .x and .y from t
    */
  SIMD.float32x4.fromFloat64x2 = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.float32x4(t.x, t.y, 0, 0);
  }
}

if (typeof SIMD.float32x4.fromInt32x4 === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @return {float32x4} An integer to float conversion copy of t.
    */
  SIMD.float32x4.fromInt32x4 = function(t) {
    t = SIMD.int32x4.check(t);
    return SIMD.float32x4(t.x, t.y, t.z, t.w);
  }
}

if (typeof SIMD.float32x4.fromFloat64x2Bits === "undefined") {
  /**
   * @param {float64x2} t An instance of float64x2.
   * @return {float32x4} a bit-wise copy of t as a float32x4.
   */
  SIMD.float32x4.fromFloat64x2Bits = function(t) {
    _SIMD_PRIVATE.saveFloat64x2(t);
    return _SIMD_PRIVATE.restoreFloat32x4();
  }
}

if (typeof SIMD.float32x4.fromInt32x4Bits === "undefined") {
  /**
   * @param {int32x4} t An instance of int32x4.
   * @return {float32x4} a bit-wise copy of t as a float32x4.
   */
  SIMD.float32x4.fromInt32x4Bits = function(t) {
    _SIMD_PRIVATE.saveInt32x4(t);
    return _SIMD_PRIVATE.restoreFloat32x4();
  }
}

if (typeof SIMD.float32x4.fromInt16x8Bits === "undefined") {
  /**
   * @param {int16x8} t An instance of int16x8.
   * @return {float32x4} a bit-wise copy of t as a float32x4.
   */
  SIMD.float32x4.fromInt16x8Bits = function(t) {
    _SIMD_PRIVATE.saveInt16x8(t);
    return _SIMD_PRIVATE.restoreFloat32x4();
  }
}

if (typeof SIMD.float32x4.fromInt8x16Bits === "undefined") {
  /**
   * @param {int8x16} t An instance of int8x16.
   * @return {float32x4} a bit-wise copy of t as a float32x4.
   */
  SIMD.float32x4.fromInt8x16Bits = function(t) {
    _SIMD_PRIVATE.saveInt8x16(t);
    return _SIMD_PRIVATE.restoreFloat32x4();
  }
}

if (typeof SIMD.float64x2 === "undefined") {
  /**
    * Construct a new instance of float64x2 number.
    * @param {double} value used for x lane.
    * @param {double} value used for y lane.
    * @constructor
    */
  SIMD.float64x2 = function(x, y) {
    if (!(this instanceof SIMD.float64x2)) {
      return new SIMD.float64x2(x, y);
    }

    // Use unary + to force coercion to Number.
    this.x_ = +x;
    this.y_ = +y;
  }

  Object.defineProperty(SIMD.float64x2.prototype, 'x', {
    get: function() { return this.x_; }
  });

  Object.defineProperty(SIMD.float64x2.prototype, 'y', {
    get: function() { return this.y_; }
  });

  /**
    * Extract the sign bit from each lane return them in the first 2 bits.
    */
  Object.defineProperty(SIMD.float64x2.prototype, 'signMask', {
    get: function() {
      var mx = (this.x < 0.0 || 1/this.x === -Infinity);
      var my = (this.y < 0.0 || 1/this.y === -Infinity);
      return mx | my << 1;
    }
  });
}

if (typeof SIMD.float64x2.check === "undefined") {
  /**
    * Check whether the argument is a float64x2.
    * @param {float64x2} v An instance of float64x2.
    * @return {float64x2} The float64x2 instance.
    */
  SIMD.float64x2.check = function(v) {
    if (!(v instanceof SIMD.float64x2)) {
      throw new TypeError("argument is not a float64x2.");
    }
    return v;
  }
}

if (typeof SIMD.float64x2.splat === "undefined") {
  /**
    * Construct a new instance of float64x2 number with the same value
    * in all lanes.
    * @param {double} value used for all lanes.
    * @constructor
    */
  SIMD.float64x2.splat = function(s) {
    return SIMD.float64x2(s, s);
  }
}

if (typeof SIMD.float64x2.fromFloat32x4 === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {float64x2} A float64x2 with .x and .y from t
    */
  SIMD.float64x2.fromFloat32x4 = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.float64x2(t.x, t.y);
  }
}

if (typeof SIMD.float64x2.fromInt32x4 === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @return {float64x2} A float64x2 with .x and .y from t
    */
  SIMD.float64x2.fromInt32x4 = function(t) {
    t = SIMD.int32x4.check(t);
    return SIMD.float64x2(t.x, t.y);
  }
}

if (typeof SIMD.float64x2.fromFloat32x4Bits === "undefined") {
  /**
   * @param {float32x4} t An instance of float32x4.
   * @return {float64x2} a bit-wise copy of t as a float64x2.
   */
  SIMD.float64x2.fromFloat32x4Bits = function(t) {
    _SIMD_PRIVATE.saveFloat32x4(t);
    return _SIMD_PRIVATE.restoreFloat64x2();
  }
}

if (typeof SIMD.float64x2.fromInt32x4Bits === "undefined") {
  /**
   * @param {int32x4} t An instance of int32x4.
   * @return {float64x2} a bit-wise copy of t as a float64x2.
   */
  SIMD.float64x2.fromInt32x4Bits = function(t) {
    _SIMD_PRIVATE.saveInt32x4(t);
    return _SIMD_PRIVATE.restoreFloat64x2();
  }
}

if (typeof SIMD.float64x2.fromInt16x8Bits === "undefined") {
  /**
   * @param {int16x8} t An instance of int16x8.
   * @return {float64x2} a bit-wise copy of t as a float64x2.
   */
  SIMD.float64x2.fromInt16x8Bits = function(t) {
    _SIMD_PRIVATE.saveInt16x8(t);
    return _SIMD_PRIVATE.restoreFloat64x2();
  }
}

if (typeof SIMD.float64x2.fromInt8x16Bits === "undefined") {
  /**
   * @param {int8x16} t An instance of int8x16.
   * @return {float64x2} a bit-wise copy of t as a float64x2.
   */
  SIMD.float64x2.fromInt8x16Bits = function(t) {
    _SIMD_PRIVATE.saveInt8x16(t);
    return _SIMD_PRIVATE.restoreFloat64x2();
  }
}

if (typeof SIMD.int32x4 === "undefined") {
  /**
    * Construct a new instance of int32x4 number.
    * @param {integer} 32-bit value used for x lane.
    * @param {integer} 32-bit value used for y lane.
    * @param {integer} 32-bit value used for z lane.
    * @param {integer} 32-bit value used for w lane.
    * @constructor
    */
  SIMD.int32x4 = function(x, y, z, w) {
    if (!(this instanceof SIMD.int32x4)) {
      return new SIMD.int32x4(x, y, z, w);
    }

    this.x_ = x|0;
    this.y_ = y|0;
    this.z_ = z|0;
    this.w_ = w|0;
  }

  Object.defineProperty(SIMD.int32x4.prototype, 'x', {
    get: function() { return this.x_; }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'y', {
    get: function() { return this.y_; }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'z', {
    get: function() { return this.z_; }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'w', {
    get: function() { return this.w_; }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'flagX', {
    get: function() { return _SIMD_PRIVATE.tobool(this.x); }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'flagY', {
    get: function() { return _SIMD_PRIVATE.tobool(this.y); }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'flagZ', {
    get: function() { return _SIMD_PRIVATE.tobool(this.z); }
  });

  Object.defineProperty(SIMD.int32x4.prototype, 'flagW', {
    get: function() { return _SIMD_PRIVATE.tobool(this.w); }
  });

  /**
    * Extract the sign bit from each lane return them in the first 4 bits.
    */
  Object.defineProperty(SIMD.int32x4.prototype, 'signMask', {
    get: function() {
      var mx = _SIMD_PRIVATE.tobool(this.x);
      var my = _SIMD_PRIVATE.tobool(this.y);
      var mz = _SIMD_PRIVATE.tobool(this.z);
      var mw = _SIMD_PRIVATE.tobool(this.w);
      return mx | my << 1 | mz << 2 | mw << 3;
    }
  });
}

if (typeof SIMD.int32x4.check === "undefined") {
  /**
    * Check whether the argument is a int32x4.
    * @param {int32x4} v An instance of int32x4.
    * @return {int32x4} The int32x4 instance.
    */
  SIMD.int32x4.check = function(v) {
    if (!(v instanceof SIMD.int32x4)) {
      throw new TypeError("argument is not a int32x4.");
    }
    return v;
  }
}

if (typeof SIMD.int32x4.bool === "undefined") {
  /**
    * Construct a new instance of int32x4 number with either true or false in each
    * lane, depending on the truth values in x, y, z, and w.
    * @param {boolean} flag used for x lane.
    * @param {boolean} flag used for y lane.
    * @param {boolean} flag used for z lane.
    * @param {boolean} flag used for w lane.
    * @constructor
    */
  SIMD.int32x4.bool = function(x, y, z, w) {
    return SIMD.int32x4(_SIMD_PRIVATE.frombool(x),
                        _SIMD_PRIVATE.frombool(y),
                        _SIMD_PRIVATE.frombool(z),
                        _SIMD_PRIVATE.frombool(w));
  }
}

if (typeof SIMD.int32x4.splat === "undefined") {
  /**
    * Construct a new instance of int32x4 number with the same value
    * in all lanes.
    * @param {integer} value used for all lanes.
    * @constructor
    */
  SIMD.int32x4.splat = function(s) {
    return SIMD.int32x4(s, s, s, s);
  }
}

if (typeof SIMD.int32x4.fromFloat32x4 === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {int32x4} with a integer to float conversion of t.
    */
  SIMD.int32x4.fromFloat32x4 = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.int32x4(t.x, t.y, t.z, t.w);
  }
}

if (typeof SIMD.int32x4.fromFloat64x2 === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @return {int32x4}  An int32x4 with .x and .y from t
    */
  SIMD.int32x4.fromFloat64x2 = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.int32x4(t.x, t.y, 0, 0);
  }
}

if (typeof SIMD.int32x4.fromFloat32x4Bits === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {int32x4} a bit-wise copy of t as a int32x4.
    */
  SIMD.int32x4.fromFloat32x4Bits = function(t) {
    _SIMD_PRIVATE.saveFloat32x4(t);
    return _SIMD_PRIVATE.restoreInt32x4();
  }
}

if (typeof SIMD.int32x4.fromFloat64x2Bits === "undefined") {
  /**
   * @param {float64x2} t An instance of float64x2.
   * @return {int32x4} a bit-wise copy of t as an int32x4.
   */
  SIMD.int32x4.fromFloat64x2Bits = function(t) {
    _SIMD_PRIVATE.saveFloat64x2(t);
    return _SIMD_PRIVATE.restoreInt32x4();
  }
}

if (typeof SIMD.int32x4.fromInt16x8Bits === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @return {int32x4} a bit-wise copy of t as a int32x4.
    */
  SIMD.int32x4.fromInt16x8Bits = function(t) {
    _SIMD_PRIVATE.saveInt16x8(t);
    return _SIMD_PRIVATE.restoreInt32x4();
  }
}

if (typeof SIMD.int32x4.fromInt8x16Bits === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @return {int32x4} a bit-wise copy of t as a int32x4.
    */
  SIMD.int32x4.fromInt8x16Bits = function(t) {
    _SIMD_PRIVATE.saveInt8x16(t);
    return _SIMD_PRIVATE.restoreInt32x4();
  }
}

if (typeof SIMD.int16x8 === "undefined") {
  /**
    * Construct a new instance of int16x8 number.
    * @param {integer} 16-bit value used for s0 lane.
    * @param {integer} 16-bit value used for s1 lane.
    * @param {integer} 16-bit value used for s2 lane.
    * @param {integer} 16-bit value used for s3 lane.
    * @param {integer} 16-bit value used for s4 lane.
    * @param {integer} 16-bit value used for s5 lane.
    * @param {integer} 16-bit value used for s6 lane.
    * @param {integer} 16-bit value used for s7 lane.
    * @constructor
    */
  SIMD.int16x8 = function(s0, s1, s2, s3, s4, s5, s6, s7) {
    if (!(this instanceof SIMD.int16x8)) {
      return new SIMD.int16x8(s0, s1, s2, s3, s4, s5, s6, s7);
    }

    this.s0_ = s0 << 16 >> 16;
    this.s1_ = s1 << 16 >> 16;
    this.s2_ = s2 << 16 >> 16;
    this.s3_ = s3 << 16 >> 16;
    this.s4_ = s4 << 16 >> 16;
    this.s5_ = s5 << 16 >> 16;
    this.s6_ = s6 << 16 >> 16;
    this.s7_ = s7 << 16 >> 16;
  }

  Object.defineProperty(SIMD.int16x8.prototype, 's0', {
    get: function() { return this.s0_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's1', {
    get: function() { return this.s1_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's2', {
    get: function() { return this.s2_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's3', {
    get: function() { return this.s3_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's4', {
    get: function() { return this.s4_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's5', {
    get: function() { return this.s5_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's6', {
    get: function() { return this.s6_; }
  });

  Object.defineProperty(SIMD.int16x8.prototype, 's7', {
    get: function() { return this.s7_; }
  });

  /**
    * Extract the sign bit from each lane return them in the first 8 bits.
    */
  Object.defineProperty(SIMD.int16x8.prototype, 'signMask', {
    get: function() {
      var ms0 = (this.s0 & 0x8000) >>> 15;
      var ms1 = (this.s1 & 0x8000) >>> 15;
      var ms2 = (this.s2 & 0x8000) >>> 15;
      var ms3 = (this.s3 & 0x8000) >>> 15;
      var ms4 = (this.s4 & 0x8000) >>> 15;
      var ms5 = (this.s5 & 0x8000) >>> 15;
      var ms6 = (this.s6 & 0x8000) >>> 15;
      var ms7 = (this.s7 & 0x8000) >>> 15;
      return ms0 | ms1 << 1 | ms2 << 2 | ms3 << 3 |
             ms4 << 4 | ms5 << 5 | ms6 << 6 | ms7 << 7;
    }
  });
}

if (typeof SIMD.int16x8.check === "undefined") {
  /**
    * Check whether the argument is a int16x8.
    * @param {int16x8} v An instance of int16x8.
    * @return {int16x8} The int16x8 instance.
    */
  SIMD.int16x8.check = function(v) {
    if (!(v instanceof SIMD.int16x8)) {
      throw new TypeError("argument is not a int16x8.");
    }
    return v;
  }
}

if (typeof SIMD.int16x8.bool === "undefined") {
  /**
    * Construct a new instance of int16x8 number with 0xFFFF or 0x0 in each
    * lane, depending on the truth value in s0, s1, s2, s3, s4, s5, s6, and s7.
    * @param {boolean} flag used for s0 lane.
    * @param {boolean} flag used for s1 lane.
    * @param {boolean} flag used for s2 lane.
    * @param {boolean} flag used for s3 lane.
    * @param {boolean} flag used for s4 lane.
    * @param {boolean} flag used for s5 lane.
    * @param {boolean} flag used for s6 lane.
    * @param {boolean} flag used for s7 lane.
    * @constructor
    */
  SIMD.int16x8.bool = function(s0, s1, s2, s3, s4, s5, s6, s7) {
    return SIMD.int16x8(s0 ? -1 : 0x0,
                        s1 ? -1 : 0x0,
                        s2 ? -1 : 0x0,
                        s3 ? -1 : 0x0,
                        s4 ? -1 : 0x0,
                        s5 ? -1 : 0x0,
                        s6 ? -1 : 0x0,
                        s7 ? -1 : 0x0);
  }
}

if (typeof SIMD.int16x8.splat === "undefined") {
  /**
    * Construct a new instance of int16x8 number with the same value
    * in all lanes.
    * @param {integer} value used for all lanes.
    * @constructor
    */
  SIMD.int16x8.splat = function(s) {
    return SIMD.int16x8(s, s, s, s, s, s, s, s);
  }
}

if (typeof SIMD.int16x8.fromFloat32x4Bits === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {int16x8} a bit-wise copy of t as a int16x8.
    */
  SIMD.int16x8.fromFloat32x4Bits = function(t) {
    _SIMD_PRIVATE.saveFloat32x4(t);
    return _SIMD_PRIVATE.restoreInt16x8();
  }
}

if (typeof SIMD.int16x8.fromFloat64x2Bits === "undefined") {
  /**
   * @param {float64x2} t An instance of float64x2.
   * @return {int16x8} a bit-wise copy of t as an int16x8.
   */
  SIMD.int16x8.fromFloat64x2Bits = function(t) {
    _SIMD_PRIVATE.saveFloat64x2(t);
    return _SIMD_PRIVATE.restoreInt16x8();
  }
}

if (typeof SIMD.int16x8.fromInt32x4Bits === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @return {int16x8} a bit-wise copy of t as a int16x8.
    */
  SIMD.int16x8.fromInt32x4Bits = function(t) {
    _SIMD_PRIVATE.saveInt32x4(t);
    return _SIMD_PRIVATE.restoreInt16x8();
  }
}

if (typeof SIMD.int16x8.fromInt8x16Bits === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @return {int16x8} a bit-wise copy of t as a int16x8.
    */
  SIMD.int16x8.fromInt8x16Bits = function(t) {
    _SIMD_PRIVATE.saveInt8x16(t);
    return _SIMD_PRIVATE.restoreInt16x8();
  }
}

if (typeof SIMD.int8x16 === "undefined") {
  /**
    * Construct a new instance of int8x16 number.
    * @param {integer} 8-bit value used for s0 lane.
    * @param {integer} 8-bit value used for s1 lane.
    * @param {integer} 8-bit value used for s2 lane.
    * @param {integer} 8-bit value used for s3 lane.
    * @param {integer} 8-bit value used for s4 lane.
    * @param {integer} 8-bit value used for s5 lane.
    * @param {integer} 8-bit value used for s6 lane.
    * @param {integer} 8-bit value used for s7 lane.
    * @param {integer} 8-bit value used for s8 lane.
    * @param {integer} 8-bit value used for s9 lane.
    * @param {integer} 8-bit value used for s10 lane.
    * @param {integer} 8-bit value used for s11 lane.
    * @param {integer} 8-bit value used for s12 lane.
    * @param {integer} 8-bit value used for s13 lane.
    * @param {integer} 8-bit value used for s14 lane.
    * @param {integer} 8-bit value used for s15 lane.
    * @constructor
    */
  SIMD.int8x16 = function(s0, s1, s2, s3, s4, s5, s6, s7,
                          s8, s9, s10, s11, s12, s13, s14, s15) {
    if (!(this instanceof SIMD.int8x16)) {
      return new SIMD.int8x16(s0, s1, s2, s3, s4, s5, s6, s7,
                              s8, s9, s10, s11, s12, s13, s14, s15);
    }

    this.s0_ = s0 << 24 >> 24;
    this.s1_ = s1 << 24 >> 24;
    this.s2_ = s2 << 24 >> 24;
    this.s3_ = s3 << 24 >> 24;
    this.s4_ = s4 << 24 >> 24;
    this.s5_ = s5 << 24 >> 24;
    this.s6_ = s6 << 24 >> 24;
    this.s7_ = s7 << 24 >> 24;
    this.s8_ = s8 << 24 >> 24;
    this.s9_ = s9 << 24 >> 24;
    this.s10_ = s10 << 24 >> 24;
    this.s11_ = s11 << 24 >> 24;
    this.s12_ = s12 << 24 >> 24;
    this.s13_ = s13 << 24 >> 24;
    this.s14_ = s14 << 24 >> 24;
    this.s15_ = s15 << 24 >> 24;
  }

  Object.defineProperty(SIMD.int8x16.prototype, 's0', {
    get: function() { return this.s0_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's1', {
    get: function() { return this.s1_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's2', {
    get: function() { return this.s2_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's3', {
    get: function() { return this.s3_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's4', {
    get: function() { return this.s4_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's5', {
    get: function() { return this.s5_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's6', {
    get: function() { return this.s6_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's7', {
    get: function() { return this.s7_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's8', {
    get: function() { return this.s8_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's9', {
    get: function() { return this.s9_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's10', {
    get: function() { return this.s10_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's11', {
    get: function() { return this.s11_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's12', {
    get: function() { return this.s12_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's13', {
    get: function() { return this.s13_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's14', {
    get: function() { return this.s14_; }
  });

  Object.defineProperty(SIMD.int8x16.prototype, 's15', {
    get: function() { return this.s15_; }
  });

  /**
    * Extract the sign bit from each lane return them in the first 16 bits.
    */
  Object.defineProperty(SIMD.int8x16.prototype, 'signMask', {
    get: function() {
      var ms0 = (this.s0 & 0x80) >>> 7;
      var ms1 = (this.s1 & 0x80) >>> 7;
      var ms2 = (this.s2 & 0x80) >>> 7;
      var ms3 = (this.s3 & 0x80) >>> 7;
      var ms4 = (this.s4 & 0x80) >>> 7;
      var ms5 = (this.s5 & 0x80) >>> 7;
      var ms6 = (this.s6 & 0x80) >>> 7;
      var ms7 = (this.s7 & 0x80) >>> 7;
      var ms8 = (this.s8 & 0x80) >>> 7;
      var ms9 = (this.s9 & 0x80) >>> 7;
      var ms10 = (this.s10 & 0x80) >>> 7;
      var ms11 = (this.s11 & 0x80) >>> 7;
      var ms12 = (this.s12 & 0x80) >>> 7;
      var ms13 = (this.s13 & 0x80) >>> 7;
      var ms14 = (this.s14 & 0x80) >>> 7;
      var ms15 = (this.s15 & 0x80) >>> 7;
      return ms0 | ms1 << 1 | ms2 << 2 | ms3 << 3 |
             ms4 << 4 | ms5 << 5 | ms6 << 6 | ms7 << 7 |
             ms8 << 8 | ms9 << 9 | ms10 << 10 | ms11 << 11 |
             ms12 << 12 | ms13 << 13 | ms14 << 14 | ms15 << 15;
    }
  });
}

if (typeof SIMD.int8x16.check === "undefined") {
  /**
    * Check whether the argument is a int8x16.
    * @param {int8x16} v An instance of int8x16.
    * @return {int8x16} The int8x16 instance.
    */
  SIMD.int8x16.check = function(v) {
    if (!(v instanceof SIMD.int8x16)) {
      throw new TypeError("argument is not a int8x16.");
    }
    return v;
  }
}

if (typeof SIMD.int8x16.bool === "undefined") {
  /**
    * Construct a new instance of int8x16 number with 0xFF or 0x0 in each
    * lane, depending on the truth value in s0, s1, s2, s3, s4, s5, s6, s7,
    * s8, s9, s10, s11, s12, s13, s14, and s15.
    * @param {boolean} flag used for s0 lane.
    * @param {boolean} flag used for s1 lane.
    * @param {boolean} flag used for s2 lane.
    * @param {boolean} flag used for s3 lane.
    * @param {boolean} flag used for s4 lane.
    * @param {boolean} flag used for s5 lane.
    * @param {boolean} flag used for s6 lane.
    * @param {boolean} flag used for s7 lane.
    * @param {boolean} flag used for s8 lane.
    * @param {boolean} flag used for s9 lane.
    * @param {boolean} flag used for s10 lane.
    * @param {boolean} flag used for s11 lane.
    * @param {boolean} flag used for s12 lane.
    * @param {boolean} flag used for s13 lane.
    * @param {boolean} flag used for s14 lane.
    * @param {boolean} flag used for s15 lane.
    * @constructor
    */
  SIMD.int8x16.bool = function(s0, s1, s2, s3, s4, s5, s6, s7,
                               s8, s9, s10, s11, s12, s13, s14, s15) {
    return SIMD.int8x16(s0 ? -1 : 0x0,
                        s1 ? -1 : 0x0,
                        s2 ? -1 : 0x0,
                        s3 ? -1 : 0x0,
                        s4 ? -1 : 0x0,
                        s5 ? -1 : 0x0,
                        s6 ? -1 : 0x0,
                        s7 ? -1 : 0x0,
                        s8 ? -1 : 0x0,
                        s9 ? -1 : 0x0,
                        s10 ? -1 : 0x0,
                        s11 ? -1 : 0x0,
                        s12 ? -1 : 0x0,
                        s13 ? -1 : 0x0,
                        s14 ? -1 : 0x0,
                        s15 ? -1 : 0x0);
  }
}

if (typeof SIMD.int8x16.splat === "undefined") {
  /**
    * Construct a new instance of int8x16 number with the same value
    * in all lanes.
    * @param {integer} value used for all lanes.
    * @constructor
    */
  SIMD.int8x16.splat = function(s) {
    return SIMD.int8x16(s, s, s, s, s, s, s, s,
                        s, s, s, s, s, s, s, s);
  }
}

if (typeof SIMD.int8x16.fromFloat32x4Bits === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {int8x16} a bit-wise copy of t as a int8x16.
    */
  SIMD.int8x16.fromFloat32x4Bits = function(t) {
    _SIMD_PRIVATE.saveFloat32x4(t);
    return _SIMD_PRIVATE.restoreInt8x16();
  }
}

if (typeof SIMD.int8x16.fromFloat64x2Bits === "undefined") {
  /**
   * @param {float64x2} t An instance of float64x2.
   * @return {int8x16} a bit-wise copy of t as an int8x16.
   */
  SIMD.int8x16.fromFloat64x2Bits = function(t) {
    _SIMD_PRIVATE.saveFloat64x2(t);
    return _SIMD_PRIVATE.restoreInt8x16();
  }
}

if (typeof SIMD.int8x16.fromInt32x4Bits === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @return {int8x16} a bit-wise copy of t as a int8x16.
    */
  SIMD.int8x16.fromInt32x4Bits = function(t) {
    _SIMD_PRIVATE.saveInt32x4(t);
    return _SIMD_PRIVATE.restoreInt8x16();
  }
}

if (typeof SIMD.int8x16.fromInt16x8Bits === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @return {int8x16} a bit-wise copy of t as a int8x16.
    */
  SIMD.int8x16.fromInt16x8Bits = function(t) {
    _SIMD_PRIVATE.saveInt16x8(t);
    return _SIMD_PRIVATE.restoreInt8x16();
  }
}

if (typeof SIMD.float32x4.abs === "undefined") {
  /**
   * @param {float32x4} t An instance of float32x4.
   * @return {float32x4} New instance of float32x4 with absolute values of
   * t.
   */
  SIMD.float32x4.abs = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.float32x4(Math.abs(t.x), Math.abs(t.y), Math.abs(t.z),
                          Math.abs(t.w));
  }
}

if (typeof SIMD.float32x4.neg === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with negated values of
    * t.
    */
  SIMD.float32x4.neg = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.float32x4(-t.x, -t.y, -t.z, -t.w);
  }
}

if (typeof SIMD.float32x4.add === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with a + b.
    */
  SIMD.float32x4.add = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    return SIMD.float32x4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}

if (typeof SIMD.float32x4.sub === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with a - b.
    */
  SIMD.float32x4.sub = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    return SIMD.float32x4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
  }
}

if (typeof SIMD.float32x4.mul === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with a * b.
    */
  SIMD.float32x4.mul = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    return SIMD.float32x4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
  }
}

if (typeof SIMD.float32x4.div === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with a / b.
    */
  SIMD.float32x4.div = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    return SIMD.float32x4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
  }
}

if (typeof SIMD.float32x4.clamp === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} lowerLimit An instance of float32x4.
    * @param {float32x4} upperLimit An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with t's values clamped
    * between lowerLimit and upperLimit.
    */
  SIMD.float32x4.clamp = function(t, lowerLimit, upperLimit) {
    t = SIMD.float32x4.check(t);
    lowerLimit = SIMD.float32x4.check(lowerLimit);
    upperLimit = SIMD.float32x4.check(upperLimit);
    var cx = t.x < lowerLimit.x ? lowerLimit.x : t.x;
    var cy = t.y < lowerLimit.y ? lowerLimit.y : t.y;
    var cz = t.z < lowerLimit.z ? lowerLimit.z : t.z;
    var cw = t.w < lowerLimit.w ? lowerLimit.w : t.w;
    cx = cx > upperLimit.x ? upperLimit.x : cx;
    cy = cy > upperLimit.y ? upperLimit.y : cy;
    cz = cz > upperLimit.z ? upperLimit.z : cz;
    cw = cw > upperLimit.w ? upperLimit.w : cw;
    return SIMD.float32x4(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.min === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with the minimum value of
    * t and other.
    */
  SIMD.float32x4.min = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = Math.min(t.x, other.x);
    var cy = Math.min(t.y, other.y);
    var cz = Math.min(t.z, other.z);
    var cw = Math.min(t.w, other.w);
    return SIMD.float32x4(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.max === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with the maximum value of
    * t and other.
    */
  SIMD.float32x4.max = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = Math.max(t.x, other.x);
    var cy = Math.max(t.y, other.y);
    var cz = Math.max(t.z, other.z);
    var cw = Math.max(t.w, other.w);
    return SIMD.float32x4(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.minNum === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with the minimum value of
    * t and other, preferring numbers over NaNs.
    */
  SIMD.float32x4.minNum = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = _SIMD_PRIVATE.minNum(t.x, other.x);
    var cy = _SIMD_PRIVATE.minNum(t.y, other.y);
    var cz = _SIMD_PRIVATE.minNum(t.z, other.z);
    var cw = _SIMD_PRIVATE.minNum(t.w, other.w);
    return SIMD.float32x4(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.maxNum === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with the maximum value of
    * t and other, preferring numbers over NaNs.
    */
  SIMD.float32x4.maxNum = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = _SIMD_PRIVATE.maxNum(t.x, other.x);
    var cy = _SIMD_PRIVATE.maxNum(t.y, other.y);
    var cz = _SIMD_PRIVATE.maxNum(t.z, other.z);
    var cw = _SIMD_PRIVATE.maxNum(t.w, other.w);
    return SIMD.float32x4(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.reciprocal === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with reciprocal value of
    * t.
    */
  SIMD.float32x4.reciprocal = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.float32x4(1.0 / t.x, 1.0 / t.y, 1.0 / t.z, 1.0 / t.w);
  }
}

if (typeof SIMD.float32x4.reciprocalSqrt === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with square root of the
    * reciprocal value of t.
    */
  SIMD.float32x4.reciprocalSqrt = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.float32x4(Math.sqrt(1.0 / t.x), Math.sqrt(1.0 / t.y),
                          Math.sqrt(1.0 / t.z), Math.sqrt(1.0 / t.w));
  }
}

if (typeof SIMD.float32x4.sqrt === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with square root of
    * values of t.
    */
  SIMD.float32x4.sqrt = function(t) {
    t = SIMD.float32x4.check(t);
    return SIMD.float32x4(Math.sqrt(t.x), Math.sqrt(t.y),
                          Math.sqrt(t.z), Math.sqrt(t.w));
  }
}

if (typeof SIMD.float32x4.swizzle === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4 to be swizzled.
    * @param {integer} x - Index in t for lane x
    * @param {integer} y - Index in t for lane y
    * @param {integer} z - Index in t for lane z
    * @param {integer} w - Index in t for lane w
    * @return {float32x4} New instance of float32x4 with lanes swizzled.
    */
  SIMD.float32x4.swizzle = function(t, x, y, z, w) {
    t = SIMD.float32x4.check(t);
    _SIMD_PRIVATE._f32x4[0] = t.x;
    _SIMD_PRIVATE._f32x4[1] = t.y;
    _SIMD_PRIVATE._f32x4[2] = t.z;
    _SIMD_PRIVATE._f32x4[3] = t.w;
    var storage = _SIMD_PRIVATE._f32x4;
    return SIMD.float32x4(storage[x], storage[y], storage[z], storage[w]);
  }
}

if (typeof SIMD.float32x4.shuffle === "undefined") {
  /**
    * @param {float32x4} t1 An instance of float32x4 to be shuffled.
    * @param {float32x4} t2 An instance of float32x4 to be shuffled.
    * @param {integer} x - Index in concatenation of t1 and t2 for lane x
    * @param {integer} y - Index in concatenation of t1 and t2 for lane y
    * @param {integer} z - Index in concatenation of t1 and t2 for lane z
    * @param {integer} w - Index in concatenation of t1 and t2 for lane w
    * @return {float32x4} New instance of float32x4 with lanes shuffled.
    */
  SIMD.float32x4.shuffle = function(t1, t2, x, y, z, w) {
    t1 = SIMD.float32x4.check(t1);
    t2 = SIMD.float32x4.check(t2);
    var storage = _SIMD_PRIVATE._f32x8;
    storage[0] = t1.x;
    storage[1] = t1.y;
    storage[2] = t1.z;
    storage[3] = t1.w;
    storage[4] = t2.x;
    storage[5] = t2.y;
    storage[6] = t2.z;
    storage[7] = t2.w;
    return SIMD.float32x4(storage[x], storage[y], storage[z], storage[w]);
  }
}

if (typeof SIMD.float32x4.withX === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {double} value used for x lane.
    * @return {float32x4} New instance of float32x4 with the values in t and
    * x replaced with {x}.
    */
  SIMD.float32x4.withX = function(t, x) {
    t = SIMD.float32x4(t);
    return SIMD.float32x4(x, t.y, t.z, t.w);
  }
}

if (typeof SIMD.float32x4.withY === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {double} value used for y lane.
    * @return {float32x4} New instance of float32x4 with the values in t and
    * y replaced with {y}.
    */
  SIMD.float32x4.withY = function(t, y) {
    t = SIMD.float32x4(t);
    return SIMD.float32x4(t.x, y, t.z, t.w);
  }
}

if (typeof SIMD.float32x4.withZ === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {double} value used for z lane.
    * @return {float32x4} New instance of float32x4 with the values in t and
    * z replaced with {z}.
    */
  SIMD.float32x4.withZ = function(t, z) {
    t = SIMD.float32x4(t);
    return SIMD.float32x4(t.x, t.y, z, t.w);
  }
}

if (typeof SIMD.float32x4.withW === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {double} value used for w lane.
    * @return {float32x4} New instance of float32x4 with the values in t and
    * w replaced with {w}.
    */
  SIMD.float32x4.withW = function(t, w) {
    t = SIMD.float32x4(t);
    return SIMD.float32x4(t.x, t.y, t.z, w);
  }
}

if (typeof SIMD.float32x4.lessThan === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t < other.
    */
  SIMD.float32x4.lessThan = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = t.x < other.x;
    var cy = t.y < other.y;
    var cz = t.z < other.z;
    var cw = t.w < other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.lessThanOrEqual === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t <= other.
    */
  SIMD.float32x4.lessThanOrEqual = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = t.x <= other.x;
    var cy = t.y <= other.y;
    var cz = t.z <= other.z;
    var cw = t.w <= other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.equal === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t == other.
    */
  SIMD.float32x4.equal = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = t.x == other.x;
    var cy = t.y == other.y;
    var cz = t.z == other.z;
    var cw = t.w == other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.notEqual === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t != other.
    */
  SIMD.float32x4.notEqual = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = t.x != other.x;
    var cy = t.y != other.y;
    var cz = t.z != other.z;
    var cw = t.w != other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.greaterThanOrEqual === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t >= other.
    */
  SIMD.float32x4.greaterThanOrEqual = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = t.x >= other.x;
    var cy = t.y >= other.y;
    var cz = t.z >= other.z;
    var cw = t.w >= other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.greaterThan === "undefined") {
  /**
    * @param {float32x4} t An instance of float32x4.
    * @param {float32x4} other An instance of float32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t > other.
    */
  SIMD.float32x4.greaterThan = function(t, other) {
    t = SIMD.float32x4.check(t);
    other = SIMD.float32x4.check(other);
    var cx = t.x > other.x;
    var cy = t.y > other.y;
    var cz = t.z > other.z;
    var cw = t.w > other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.float32x4.select === "undefined") {
  /**
    * @param {int32x4} t Selector mask. An instance of int32x4
    * @param {float32x4} trueValue Pick lane from here if corresponding
    * selector lane is true
    * @param {float32x4} falseValue Pick lane from here if corresponding
    * selector lane is false
    * @return {float32x4} Mix of lanes from trueValue or falseValue as
    * indicated
    */
  SIMD.float32x4.select = function(t, trueValue, falseValue) {
    t = SIMD.int32x4.check(t);
    trueValue = SIMD.float32x4.check(trueValue);
    falseValue = SIMD.float32x4.check(falseValue);
    return SIMD.float32x4(_SIMD_PRIVATE.tobool(t.x) ? trueValue.x : falseValue.x,
                          _SIMD_PRIVATE.tobool(t.y) ? trueValue.y : falseValue.y,
                          _SIMD_PRIVATE.tobool(t.z) ? trueValue.z : falseValue.z,
                          _SIMD_PRIVATE.tobool(t.w) ? trueValue.w : falseValue.w);
  }
}

if (typeof SIMD.float32x4.bitselect === "undefined") {
  /**
    * @param {int32x4} t Selector mask. An instance of int32x4
    * @param {float32x4} trueValue Pick bit from here if corresponding
    * selector bit is 1
    * @param {float32x4} falseValue Pick bit from here if corresponding
    * selector bit is 0
    * @return {float32x4} Mix of bits from trueValue or falseValue as
    * indicated
    */
  SIMD.float32x4.bitselect = function(t, trueValue, falseValue) {
    t = SIMD.int32x4.check(t);
    trueValue = SIMD.float32x4.check(trueValue);
    falseValue = SIMD.float32x4.check(falseValue);
    var tv = SIMD.int32x4.fromFloat32x4Bits(trueValue);
    var fv = SIMD.int32x4.fromFloat32x4Bits(falseValue);
    var tr = SIMD.int32x4.and(t, tv);
    var fr = SIMD.int32x4.and(SIMD.int32x4.not(t), fv);
    return SIMD.float32x4.fromInt32x4Bits(SIMD.int32x4.or(tr, fr));
  }
}

if (typeof SIMD.float32x4.and === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with values of a & b.
    */
  SIMD.float32x4.and = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    var aInt = SIMD.int32x4.fromFloat32x4Bits(a);
    var bInt = SIMD.int32x4.fromFloat32x4Bits(b);
    return SIMD.float32x4.fromInt32x4Bits(SIMD.int32x4.and(aInt, bInt));
  }
}

if (typeof SIMD.float32x4.or === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with values of a | b.
    */
  SIMD.float32x4.or = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    var aInt = SIMD.int32x4.fromFloat32x4Bits(a);
    var bInt = SIMD.int32x4.fromFloat32x4Bits(b);
    return SIMD.float32x4.fromInt32x4Bits(SIMD.int32x4.or(aInt, bInt));
  }
}

if (typeof SIMD.float32x4.xor === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @param {float32x4} b An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with values of a ^ b.
    */
  SIMD.float32x4.xor = function(a, b) {
    a = SIMD.float32x4.check(a);
    b = SIMD.float32x4.check(b);
    var aInt = SIMD.int32x4.fromFloat32x4Bits(a);
    var bInt = SIMD.int32x4.fromFloat32x4Bits(b);
    return SIMD.float32x4.fromInt32x4Bits(SIMD.int32x4.xor(aInt, bInt));
  }
}

if (typeof SIMD.float32x4.not === "undefined") {
  /**
    * @param {float32x4} a An instance of float32x4.
    * @return {float32x4} New instance of float32x4 with values of ~a.
    */
  SIMD.float32x4.not = function(a) {
    a = SIMD.float32x4.check(a);
    var aInt = SIMD.int32x4.fromFloat32x4Bits(a);
    return SIMD.float32x4.fromInt32x4Bits(SIMD.int32x4.not(aInt));
  }
}

if (typeof SIMD.float32x4.load === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {float32x4} New instance of float32x4.
    */
  SIMD.float32x4.load = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var f32temp = _SIMD_PRIVATE._f32x4;
    return SIMD.float32x4(f32temp[0], f32temp[1], f32temp[2], f32temp[3]);
  }
}

if (typeof SIMD.float32x4.loadX === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {float32x4} New instance of float32x4.
    */
  SIMD.float32x4.loadX = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 4) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 4 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var f32temp = _SIMD_PRIVATE._f32x4;
    return SIMD.float32x4(f32temp[0], 0.0, 0.0, 0.0);
  }
}

if (typeof SIMD.float32x4.loadXY === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {float32x4} New instance of float32x4.
    */
  SIMD.float32x4.loadXY = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 8) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 8 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var f32temp = _SIMD_PRIVATE._f32x4;
    return SIMD.float32x4(f32temp[0], f32temp[1], 0.0, 0.0);
  }
}

if (typeof SIMD.float32x4.loadXYZ === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {float32x4} New instance of float32x4.
    */
  SIMD.float32x4.loadXYZ = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 12) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 12 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var f32temp = _SIMD_PRIVATE._f32x4;
    return SIMD.float32x4(f32temp[0], f32temp[1], f32temp[2], 0.0);
  }
}

if (typeof SIMD.float32x4.store === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {float32x4} value An instance of float32x4.
    * @return {void}
    */
  SIMD.float32x4.store = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.float32x4.check(value);
    _SIMD_PRIVATE._f32x4[0] = value.x;
    _SIMD_PRIVATE._f32x4[1] = value.y;
    _SIMD_PRIVATE._f32x4[2] = value.z;
    _SIMD_PRIVATE._f32x4[3] = value.w;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.float32x4.storeX === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {float32x4} value An instance of float32x4.
    * @return {void}
    */
  SIMD.float32x4.storeX = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 4) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.float32x4.check(value);
    if (bpe == 8) {
      // tarray's elements are too wide. Just create a new view; this is rare.
      var view = new Float32Array(tarray.buffer, tarray.byteOffset + index * 8, 1);
      view[0] = value.x;
    } else {
      _SIMD_PRIVATE._f32x4[0] = value.x;
      var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                  bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                  (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4);
      var n = 4 / bpe;
      for (var i = 0; i < n; ++i)
        tarray[index + i] = array[i];
    }
  }
}

if (typeof SIMD.float32x4.storeXY === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {float32x4} value An instance of float32x4.
    * @return {void}
    */
  SIMD.float32x4.storeXY = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 8) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.float32x4.check(value);
    _SIMD_PRIVATE._f32x4[0] = value.x;
    _SIMD_PRIVATE._f32x4[1] = value.y;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 8 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.float32x4.storeXYZ === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {float32x4} value An instance of float32x4.
    * @return {void}
    */
  SIMD.float32x4.storeXYZ = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 12) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.float32x4.check(value);
    if (bpe == 8) {
      // tarray's elements are too wide. Just create a new view; this is rare.
      var view = new Float32Array(tarray.buffer, tarray.byteOffset + index * 8, 3);
      view[0] = value.x;
      view[1] = value.y;
      view[2] = value.z;
    } else {
      _SIMD_PRIVATE._f32x4[0] = value.x;
      _SIMD_PRIVATE._f32x4[1] = value.y;
      _SIMD_PRIVATE._f32x4[2] = value.z;
      var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                  bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                  (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4);
      var n = 12 / bpe;
      for (var i = 0; i < n; ++i)
        tarray[index + i] = array[i];
    }
  }
}

if (typeof SIMD.float64x2.abs === "undefined") {
  /**
   * @param {float64x2} t An instance of float64x2.
   * @return {float64x2} New instance of float64x2 with absolute values of
   * t.
   */
  SIMD.float64x2.abs = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.float64x2(Math.abs(t.x), Math.abs(t.y));
  }
}

if (typeof SIMD.float64x2.neg === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with negated values of
    * t.
    */
  SIMD.float64x2.neg = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.float64x2(-t.x, -t.y);
  }
}

if (typeof SIMD.float64x2.add === "undefined") {
  /**
    * @param {float64x2} a An instance of float64x2.
    * @param {float64x2} b An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with a + b.
    */
  SIMD.float64x2.add = function(a, b) {
    a = SIMD.float64x2.check(a);
    b = SIMD.float64x2.check(b);
    return SIMD.float64x2(a.x + b.x, a.y + b.y);
  }
}

if (typeof SIMD.float64x2.sub === "undefined") {
  /**
    * @param {float64x2} a An instance of float64x2.
    * @param {float64x2} b An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with a - b.
    */
  SIMD.float64x2.sub = function(a, b) {
    a = SIMD.float64x2.check(a);
    b = SIMD.float64x2.check(b);
    return SIMD.float64x2(a.x - b.x, a.y - b.y);
  }
}

if (typeof SIMD.float64x2.mul === "undefined") {
  /**
    * @param {float64x2} a An instance of float64x2.
    * @param {float64x2} b An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with a * b.
    */
  SIMD.float64x2.mul = function(a, b) {
    a = SIMD.float64x2.check(a);
    b = SIMD.float64x2.check(b);
    return SIMD.float64x2(a.x * b.x, a.y * b.y);
  }
}

if (typeof SIMD.float64x2.div === "undefined") {
  /**
    * @param {float64x2} a An instance of float64x2.
    * @param {float64x2} b An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with a / b.
    */
  SIMD.float64x2.div = function(a, b) {
    a = SIMD.float64x2.check(a);
    b = SIMD.float64x2.check(b);
    return SIMD.float64x2(a.x / b.x, a.y / b.y);
  }
}

if (typeof SIMD.float64x2.clamp === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} lowerLimit An instance of float64x2.
    * @param {float64x2} upperLimit An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with t's values clamped
    * between lowerLimit and upperLimit.
    */
  SIMD.float64x2.clamp = function(t, lowerLimit, upperLimit) {
    t = SIMD.float64x2.check(t);
    lowerLimit = SIMD.float64x2.check(lowerLimit);
    upperLimit = SIMD.float64x2.check(upperLimit);
    var cx = t.x < lowerLimit.x ? lowerLimit.x : t.x;
    var cy = t.y < lowerLimit.y ? lowerLimit.y : t.y;
    cx = cx > upperLimit.x ? upperLimit.x : cx;
    cy = cy > upperLimit.y ? upperLimit.y : cy;
    return SIMD.float64x2(cx, cy);
  }
}

if (typeof SIMD.float64x2.min === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with the minimum value of
    * t and other.
    */
  SIMD.float64x2.min = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = Math.min(t.x, other.x);
    var cy = Math.min(t.y, other.y);
    return SIMD.float64x2(cx, cy);
  }
}

if (typeof SIMD.float64x2.max === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with the maximum value of
    * t and other.
    */
  SIMD.float64x2.max = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = Math.max(t.x, other.x);
    var cy = Math.max(t.y, other.y);
    return SIMD.float64x2(cx, cy);
  }
}

if (typeof SIMD.float64x2.minNum === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with the minimum value of
    * t and other, preferring numbers over NaNs.
    */
  SIMD.float64x2.minNum = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = _SIMD_PRIVATE.minNum(t.x, other.x);
    var cy = _SIMD_PRIVATE.minNum(t.y, other.y);
    return SIMD.float64x2(cx, cy);
  }
}

if (typeof SIMD.float64x2.maxNum === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with the maximum value of
    * t and other, preferring numbers over NaNs.
    */
  SIMD.float64x2.maxNum = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = _SIMD_PRIVATE.maxNum(t.x, other.x);
    var cy = _SIMD_PRIVATE.maxNum(t.y, other.y);
    return SIMD.float64x2(cx, cy);
  }
}

if (typeof SIMD.float64x2.reciprocal === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with reciprocal value of
    * t.
    */
  SIMD.float64x2.reciprocal = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.float64x2(1.0 / t.x, 1.0 / t.y);
  }
}

if (typeof SIMD.float64x2.reciprocalSqrt === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with square root of the
    * reciprocal value of t.
    */
  SIMD.float64x2.reciprocalSqrt = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.float64x2(Math.sqrt(1.0 / t.x), Math.sqrt(1.0 / t.y));
  }
}

if (typeof SIMD.float64x2.sqrt === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @return {float64x2} New instance of float64x2 with square root of
    * values of t.
    */
  SIMD.float64x2.sqrt = function(t) {
    t = SIMD.float64x2.check(t);
    return SIMD.float64x2(Math.sqrt(t.x), Math.sqrt(t.y));
  }
}

if (typeof SIMD.float64x2.swizzle === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2 to be swizzled.
    * @param {integer} x - Index in t for lane x
    * @param {integer} y - Index in t for lane y
    * @return {float64x2} New instance of float64x2 with lanes swizzled.
    */
  SIMD.float64x2.swizzle = function(t, x, y) {
    t = SIMD.float64x2.check(t);
    var storage = _SIMD_PRIVATE._f64x2;
    storage[0] = t.x;
    storage[1] = t.y;
    return SIMD.float64x2(storage[x], storage[y]);
  }
}

if (typeof SIMD.float64x2.shuffle === "undefined") {
  /**
    * @param {float64x2} t1 An instance of float64x2 to be shuffled.
    * @param {float64x2} t2 An instance of float64x2 to be shuffled.
    * @param {integer} x - Index in concatenation of t1 and t2 for lane x
    * @param {integer} y - Index in concatenation of t1 and t2 for lane y
    * @return {float64x2} New instance of float64x2 with lanes shuffled.
    */
  SIMD.float64x2.shuffle = function(t1, t2, x, y) {
    t1 = SIMD.float64x2.check(t1);
    t2 = SIMD.float64x2.check(t2);
    var storage = _SIMD_PRIVATE._f64x4;
    storage[0] = t1.x;
    storage[1] = t1.y;
    storage[2] = t2.x;
    storage[3] = t2.y;
    return SIMD.float64x2(storage[x], storage[y]);
  }
}

if (typeof SIMD.float64x2.withX === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {double} value used for x lane.
    * @return {float64x2} New instance of float64x2 with the values in t and
    * x replaced with {x}.
    */
  SIMD.float64x2.withX = function(t, x) {
    t = SIMD.float64x2(t);
    return SIMD.float64x2(x, t.y);
  }
}

if (typeof SIMD.float64x2.withY === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {double} value used for y lane.
    * @return {float64x2} New instance of float64x2 with the values in t and
    * y replaced with {y}.
    */
  SIMD.float64x2.withY = function(t, y) {
    t = SIMD.float64x2(t);
    return SIMD.float64x2(t.x, y);
  }
}

if (typeof SIMD.float64x2.lessThan === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {int32x4} true or false in each lane depending on
    * the result of t < other.
    */
  SIMD.float64x2.lessThan = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = t.x < other.x;
    var cy = t.y < other.y;
    return SIMD.int32x4.bool(cx, cx, cy, cy);
  }
}

if (typeof SIMD.float64x2.lessThanOrEqual === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {int32x4} true or false in each lane depending on
    * the result of t <= other.
    */
  SIMD.float64x2.lessThanOrEqual = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = t.x <= other.x;
    var cy = t.y <= other.y;
    return SIMD.int32x4.bool(cx, cx, cy, cy);
  }
}

if (typeof SIMD.float64x2.equal === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {int32x4} true or false in each lane depending on
    * the result of t == other.
    */
  SIMD.float64x2.equal = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = t.x == other.x;
    var cy = t.y == other.y;
    return SIMD.int32x4.bool(cx, cx, cy, cy);
  }
}

if (typeof SIMD.float64x2.notEqual === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {int32x4} true or false in each lane depending on
    * the result of t != other.
    */
  SIMD.float64x2.notEqual = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = t.x != other.x;
    var cy = t.y != other.y;
    return SIMD.int32x4.bool(cx, cx, cy, cy);
  }
}

if (typeof SIMD.float64x2.greaterThanOrEqual === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {int32x4} true or false in each lane depending on
    * the result of t >= other.
    */
  SIMD.float64x2.greaterThanOrEqual = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = t.x >= other.x;
    var cy = t.y >= other.y;
    return SIMD.int32x4.bool(cx, cx, cy, cy);
  }
}

if (typeof SIMD.float64x2.greaterThan === "undefined") {
  /**
    * @param {float64x2} t An instance of float64x2.
    * @param {float64x2} other An instance of float64x2.
    * @return {int32x4} true or false in each lane depending on
    * the result of t > other.
    */
  SIMD.float64x2.greaterThan = function(t, other) {
    t = SIMD.float64x2.check(t);
    other = SIMD.float64x2.check(other);
    var cx = t.x > other.x;
    var cy = t.y > other.y;
    return SIMD.int32x4.bool(cx, cx, cy, cy);
  }
}

if (typeof SIMD.float64x2.select === "undefined") {
  /**
    * @param {int32x4} t Selector mask. An instance of int32x4
    * @param {float64x2} trueValue Pick lane from here if corresponding
    * selector lane is true
    * @param {float64x2} falseValue Pick lane from here if corresponding
    * selector lane is false
    * @return {float64x2} Mix of lanes from trueValue or falseValue as
    * indicated
    */
  SIMD.float64x2.select = function(t, trueValue, falseValue) {
    t = SIMD.int32x4.check(t);
    trueValue = SIMD.float64x2.check(trueValue);
    falseValue = SIMD.float64x2.check(falseValue);
    return SIMD.float64x2(_SIMD_PRIVATE.tobool(t.x) ? trueValue.x : falseValue.x,
                          _SIMD_PRIVATE.tobool(t.y) ? trueValue.y : falseValue.y);
  }
}

if (typeof SIMD.float64x2.bitselect === "undefined") {
  /**
    * @param {int32x4} t Selector mask. An instance of int32x4
    * @param {float64x2} trueValue Pick bit from here if corresponding
    * selector bit is 1
    * @param {float64x2} falseValue Pick bit from here if corresponding
    * selector bit is 0
    * @return {float64x2} Mix of bits from trueValue or falseValue as
    * indicated
    */
  SIMD.float64x2.bitselect = function(t, trueValue, falseValue) {
    t = SIMD.int32x4.check(t);
    trueValue = SIMD.float64x2.check(trueValue);
    falseValue = SIMD.float64x2.check(falseValue);
    var tv = SIMD.int32x4.fromFloat64x2Bits(trueValue);
    var fv = SIMD.int32x4.fromFloat64x2Bits(falseValue);
    var tr = SIMD.int32x4.and(t, tv);
    var fr = SIMD.int32x4.and(SIMD.int32x4.not(t), fv);
    return SIMD.float64x2.fromInt32x4Bits(SIMD.int32x4.or(tr, fr));
  }
}

if (typeof SIMD.float64x2.load === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {float64x2} New instance of float64x2.
    */
  SIMD.float64x2.load = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var f64temp = _SIMD_PRIVATE._f64x2;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                f64temp;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    return SIMD.float64x2(f64temp[0], f64temp[1]);
  }
}

if (typeof SIMD.float64x2.loadX === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {float64x2} New instance of float64x2.
    */
  SIMD.float64x2.loadX = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 8) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var f64temp = _SIMD_PRIVATE._f64x2;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                f64temp;
    var n = 8 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    return SIMD.float64x2(f64temp[0], 0.0);
  }
}

if (typeof SIMD.float64x2.store === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {float64x2} value An instance of float64x2.
    * @return {void}
    */
  SIMD.float64x2.store = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.float64x2.check(value);
    _SIMD_PRIVATE._f64x2[0] = value.x;
    _SIMD_PRIVATE._f64x2[1] = value.y;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.float64x2.storeX === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {float64x2} value An instance of float64x2.
    * @return {void}
    */
  SIMD.float64x2.storeX = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 8) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.float64x2.check(value);
    _SIMD_PRIVATE._f64x2[0] = value.x;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 8 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.int32x4.and === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {int32x4} b An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of a & b.
    */
  SIMD.int32x4.and = function(a, b) {
    a = SIMD.int32x4.check(a);
    b = SIMD.int32x4.check(b);
    return SIMD.int32x4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
  }
}

if (typeof SIMD.int32x4.or === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {int32x4} b An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of a | b.
    */
  SIMD.int32x4.or = function(a, b) {
    a = SIMD.int32x4.check(a);
    b = SIMD.int32x4.check(b);
    return SIMD.int32x4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
  }
}

if (typeof SIMD.int32x4.xor === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {int32x4} b An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of a ^ b.
    */
  SIMD.int32x4.xor = function(a, b) {
    a = SIMD.int32x4.check(a);
    b = SIMD.int32x4.check(b);
    return SIMD.int32x4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
  }
}

if (typeof SIMD.int32x4.not === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of ~t
    */
  SIMD.int32x4.not = function(t) {
    t = SIMD.int32x4.check(t);
    return SIMD.int32x4(~t.x, ~t.y, ~t.z, ~t.w);
  }
}

if (typeof SIMD.int32x4.neg === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of -t
    */
  SIMD.int32x4.neg = function(t) {
    t = SIMD.int32x4.check(t);
    return SIMD.int32x4(-t.x, -t.y, -t.z, -t.w);
  }
}

if (typeof SIMD.int32x4.add === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {int32x4} b An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of a + b.
    */
  SIMD.int32x4.add = function(a, b) {
    a = SIMD.int32x4.check(a);
    b = SIMD.int32x4.check(b);
    return SIMD.int32x4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}

if (typeof SIMD.int32x4.sub === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {int32x4} b An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of a - b.
    */
  SIMD.int32x4.sub = function(a, b) {
    a = SIMD.int32x4.check(a);
    b = SIMD.int32x4.check(b);
    return SIMD.int32x4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
  }
}

if (typeof SIMD.int32x4.mul === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {int32x4} b An instance of int32x4.
    * @return {int32x4} New instance of int32x4 with values of a * b.
    */
  SIMD.int32x4.mul = function(a, b) {
    a = SIMD.int32x4.check(a);
    b = SIMD.int32x4.check(b);
    return SIMD.int32x4(Math.imul(a.x, b.x), Math.imul(a.y, b.y),
                        Math.imul(a.z, b.z), Math.imul(a.w, b.w));
  }
}

if (typeof SIMD.int32x4.swizzle === "undefined") {
  /**
    * @param {int32x4} t An instance of float32x4 to be swizzled.
    * @param {integer} x - Index in t for lane x
    * @param {integer} y - Index in t for lane y
    * @param {integer} z - Index in t for lane z
    * @param {integer} w - Index in t for lane w
    * @return {int32x4} New instance of float32x4 with lanes swizzled.
    */
  SIMD.int32x4.swizzle = function(t, x, y, z, w) {
    t = SIMD.int32x4.check(t);
    var storage = _SIMD_PRIVATE._i32x4;
    storage[0] = t.x;
    storage[1] = t.y;
    storage[2] = t.z;
    storage[3] = t.w;
    return SIMD.int32x4(storage[x], storage[y], storage[z], storage[w]);
  }
}

if (typeof SIMD.int32x4.shuffle === "undefined") {
  /**
    * @param {int32x4} t1 An instance of float32x4 to be shuffled.
    * @param {int32x4} t2 An instance of float32x4 to be shuffled.
    * @param {integer} x - Index in concatenation of t1 and t2 for lane x
    * @param {integer} y - Index in concatenation of t1 and t2 for lane y
    * @param {integer} z - Index in concatenation of t1 and t2 for lane z
    * @param {integer} w - Index in concatenation of t1 and t2 for lane w
    * @return {int32x4} New instance of float32x4 with lanes shuffled.
    */
  SIMD.int32x4.shuffle = function(t1, t2, x, y, z, w) {
    t1 = SIMD.int32x4.check(t1);
    t2 = SIMD.int32x4.check(t2);
    var storage = _SIMD_PRIVATE._i32x8;
    storage[0] = t1.x;
    storage[1] = t1.y;
    storage[2] = t1.z;
    storage[3] = t1.w;
    storage[4] = t2.x;
    storage[5] = t2.y;
    storage[6] = t2.z;
    storage[7] = t2.w;
    return SIMD.float32x4(storage[x], storage[y], storage[z], storage[w]);
  }
}

if (typeof SIMD.int32x4.select === "undefined") {
  /**
    * @param {int32x4} t Selector mask. An instance of int32x4
    * @param {int32x4} trueValue Pick lane from here if corresponding
    * selector lane is true
    * @param {int32x4} falseValue Pick lane from here if corresponding
    * selector lane is false
    * @return {int32x4} Mix of lanes from trueValue or falseValue as
    * indicated
    */
  SIMD.int32x4.select = function(t, trueValue, falseValue) {
    t = SIMD.int32x4.check(t);
    trueValue = SIMD.int32x4.check(trueValue);
    falseValue = SIMD.int32x4.check(falseValue);
    return SIMD.int32x4(_SIMD_PRIVATE.tobool(t.x) ? trueValue.x : falseValue.x,
                        _SIMD_PRIVATE.tobool(t.y) ? trueValue.y : falseValue.y,
                        _SIMD_PRIVATE.tobool(t.z) ? trueValue.z : falseValue.z,
                        _SIMD_PRIVATE.tobool(t.w) ? trueValue.w : falseValue.w);
  }
}

if (typeof SIMD.int32x4.bitselect === "undefined") {
  /**
    * @param {int32x4} t Selector mask. An instance of int32x4
    * @param {int32x4} trueValue Pick bit from here if corresponding
    * selector bit is 1
    * @param {int32x4} falseValue Pick bit from here if corresponding
    * selector bit is 0
    * @return {int32x4} Mix of bits from trueValue or falseValue as
    * indicated
    */
  SIMD.int32x4.bitselect = function(t, trueValue, falseValue) {
    t = SIMD.int32x4.check(t);
    trueValue = SIMD.int32x4.check(trueValue);
    falseValue = SIMD.int32x4.check(falseValue);
    var tr = SIMD.int32x4.and(t, trueValue);
    var fr = SIMD.int32x4.and(SIMD.int32x4.not(t), falseValue);
    return SIMD.int32x4.or(tr, fr);
  }
}

if (typeof SIMD.int32x4.withX === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {integer} 32-bit value used for x lane.
    * @return {int32x4} New instance of int32x4 with the values in t and
    * x lane replaced with {x}.
    */
  SIMD.int32x4.withX = function(t, x) {
    t = SIMD.int32x4(t);
    return SIMD.int32x4(x, t.y, t.z, t.w);
  }
}

if (typeof SIMD.int32x4.withY === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {integer} 32-bit value used for y lane.
    * @return {int32x4} New instance of int32x4 with the values in t and
    * y lane replaced with {y}.
    */
  SIMD.int32x4.withY = function(t, y) {
    t = SIMD.int32x4(t);
    return SIMD.int32x4(t.x, y, t.z, t.w);
  }
}

if (typeof SIMD.int32x4.withZ === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {integer} 32-bit value used for z lane.
    * @return {int32x4} New instance of int32x4 with the values in t and
    * z lane replaced with {z}.
    */
  SIMD.int32x4.withZ = function(t, z) {
    t = SIMD.int32x4(t);
    return SIMD.int32x4(t.x, t.y, z, t.w);
  }
}

if (typeof SIMD.int32x4.withW === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {integer} 32-bit value used for w lane.
    * @return {int32x4} New instance of int32x4 with the values in t and
    * w lane replaced with {w}.
    */
  SIMD.int32x4.withW = function(t, w) {
    t = SIMD.int32x4(t);
    return SIMD.int32x4(t.x, t.y, t.z, w);
  }
}

if (typeof SIMD.int32x4.equal === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {int32x4} other An instance of int32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t == other.
    */
  SIMD.int32x4.equal = function(t, other) {
    t = SIMD.int32x4.check(t);
    other = SIMD.int32x4.check(other);
    var cx = t.x == other.x;
    var cy = t.y == other.y;
    var cz = t.z == other.z;
    var cw = t.w == other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.int32x4.notEqual === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {int32x4} other An instance of int32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t != other.
    */
  SIMD.int32x4.notEqual = function(t, other) {
    t = SIMD.int32x4.check(t);
    other = SIMD.int32x4.check(other);
    var cx = t.x != other.x;
    var cy = t.y != other.y;
    var cz = t.z != other.z;
    var cw = t.w != other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.int32x4.greaterThan === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {int32x4} other An instance of int32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t > other.
    */
  SIMD.int32x4.greaterThan = function(t, other) {
    t = SIMD.int32x4.check(t);
    other = SIMD.int32x4.check(other);
    var cx = t.x > other.x;
    var cy = t.y > other.y;
    var cz = t.z > other.z;
    var cw = t.w > other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.int32x4.greaterThanOrEqual === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {int32x4} other An instance of int32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t >= other.
    */
  SIMD.int32x4.greaterThanOrEqual = function(t, other) {
    t = SIMD.int32x4.check(t);
    other = SIMD.int32x4.check(other);
    var cx = t.x >= other.x;
    var cy = t.y >= other.y;
    var cz = t.z >= other.z;
    var cw = t.w >= other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.int32x4.lessThan === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {int32x4} other An instance of int32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t < other.
    */
  SIMD.int32x4.lessThan = function(t, other) {
    t = SIMD.int32x4.check(t);
    other = SIMD.int32x4.check(other);
    var cx = t.x < other.x;
    var cy = t.y < other.y;
    var cz = t.z < other.z;
    var cw = t.w < other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.int32x4.lessThanOrEqual === "undefined") {
  /**
    * @param {int32x4} t An instance of int32x4.
    * @param {int32x4} other An instance of int32x4.
    * @return {int32x4} true or false in each lane depending on
    * the result of t <= other.
    */
  SIMD.int32x4.lessThanOrEqual = function(t, other) {
    t = SIMD.int32x4.check(t);
    other = SIMD.int32x4.check(other);
    var cx = t.x <= other.x;
    var cy = t.y <= other.y;
    var cz = t.z <= other.z;
    var cw = t.w <= other.w;
    return SIMD.int32x4.bool(cx, cy, cz, cw);
  }
}

if (typeof SIMD.int32x4.shiftLeftByScalar === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {integer} bits Bit count to shift by.
    * @return {int32x4} lanes in a shifted by bits.
    */
  SIMD.int32x4.shiftLeftByScalar = function(a, bits) {
    a = SIMD.int32x4.check(a);
    var x = a.x << bits;
    var y = a.y << bits;
    var z = a.z << bits;
    var w = a.w << bits;
    return SIMD.int32x4(x, y, z, w);
  }
}

if (typeof SIMD.int32x4.shiftRightLogicalByScalar === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {integer} bits Bit count to shift by.
    * @return {int32x4} lanes in a shifted by bits.
    */
  SIMD.int32x4.shiftRightLogicalByScalar = function(a, bits) {
    a = SIMD.int32x4.check(a);
    var x = a.x >>> bits;
    var y = a.y >>> bits;
    var z = a.z >>> bits;
    var w = a.w >>> bits;
    return SIMD.int32x4(x, y, z, w);
  }
}

if (typeof SIMD.int32x4.shiftRightArithmeticByScalar === "undefined") {
  /**
    * @param {int32x4} a An instance of int32x4.
    * @param {integer} bits Bit count to shift by.
    * @return {int32x4} lanes in a shifted by bits.
    */
  SIMD.int32x4.shiftRightArithmeticByScalar = function(a, bits) {
    a = SIMD.int32x4.check(a);
    var x = a.x >> bits;
    var y = a.y >> bits;
    var z = a.z >> bits;
    var w = a.w >> bits;
    return SIMD.int32x4(x, y, z, w);
  }
}

if (typeof SIMD.int32x4.load === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {int32x4} New instance of int32x4.
    */
  SIMD.int32x4.load = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var i32temp = _SIMD_PRIVATE._i32x4;
    return SIMD.int32x4(i32temp[0], i32temp[1], i32temp[2], i32temp[3]);
  }
}

if (typeof SIMD.int32x4.loadX === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {int32x4} New instance of int32x4.
    */
  SIMD.int32x4.loadX = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 4) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 4 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var i32temp = _SIMD_PRIVATE._i32x4;
    return SIMD.int32x4(i32temp[0], 0, 0, 0);
  }
}

if (typeof SIMD.int32x4.loadXY === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {int32x4} New instance of int32x4.
    */
  SIMD.int32x4.loadXY = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 8) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 8 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var i32temp = _SIMD_PRIVATE._i32x4;
    return SIMD.int32x4(i32temp[0], i32temp[1], 0, 0);
  }
}

if (typeof SIMD.int32x4.loadXYZ === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {int32x4} New instance of int32x4.
    */
  SIMD.int32x4.loadXYZ = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 12) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 12 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    var i32temp = _SIMD_PRIVATE._i32x4;
    return SIMD.int32x4(i32temp[0], i32temp[1], i32temp[2], 0);
  }
}

if (typeof SIMD.int32x4.store === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {int32x4} value An instance of int32x4.
    * @return {void}
    */
  SIMD.int32x4.store = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.int32x4.check(value);
    _SIMD_PRIVATE._i32x4[0] = value.x;
    _SIMD_PRIVATE._i32x4[1] = value.y;
    _SIMD_PRIVATE._i32x4[2] = value.z;
    _SIMD_PRIVATE._i32x4[3] = value.w;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.int32x4.storeX === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {int32x4} value An instance of int32x4.
    * @return {void}
    */
  SIMD.int32x4.storeX = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 4) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.int32x4.check(value);
    if (bpe == 8) {
      // tarray's elements are too wide. Just create a new view; this is rare.
      var view = new Int32Array(tarray.buffer, tarray.byteOffset + index * 8, 1);
      view[0] = value.x;
    } else {
      _SIMD_PRIVATE._i32x4[0] = value.x;
      var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                  bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                  (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4);
      var n = 4 / bpe;
      for (var i = 0; i < n; ++i)
        tarray[index + i] = array[i];
    }
  }
}

if (typeof SIMD.int32x4.storeXY === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {int32x4} value An instance of int32x4.
    * @return {void}
    */
  SIMD.int32x4.storeXY = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 8) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.int32x4.check(value);
    _SIMD_PRIVATE._i32x4[0] = value.x;
    _SIMD_PRIVATE._i32x4[1] = value.y;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 8 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.int32x4.storeXYZ === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {int32x4} value An instance of int32x4.
    * @return {void}
    */
  SIMD.int32x4.storeXYZ = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 12) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.int32x4.check(value);
    if (bpe == 8) {
      // tarray's elements are too wide. Just create a new view; this is rare.
      var view = new Int32Array(tarray.buffer, tarray.byteOffset + index * 8, 3);
      view[0] = value.x;
      view[1] = value.y;
      view[2] = value.z;
    } else {
      _SIMD_PRIVATE._i32x4[0] = value.x;
      _SIMD_PRIVATE._i32x4[1] = value.y;
      _SIMD_PRIVATE._i32x4[2] = value.z;
      var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                  bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                  (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4);
      var n = 12 / bpe;
      for (var i = 0; i < n; ++i)
        tarray[index + i] = array[i];
    }
  }
}

if (typeof SIMD.int16x8.and === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {int16x8} b An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of a & b.
    */
  SIMD.int16x8.and = function(a, b) {
    a = SIMD.int16x8.check(a);
    b = SIMD.int16x8.check(b);
    return SIMD.int16x8(a.s0 & b.s0, a.s1 & b.s1, a.s2 & b.s2, a.s3 & b.s3,
                        a.s4 & b.s4, a.s5 & b.s5, a.s6 & b.s6, a.s7 & b.s7);
  }
}

if (typeof SIMD.int16x8.or === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {int16x8} b An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of a | b.
    */
  SIMD.int16x8.or = function(a, b) {
    a = SIMD.int16x8.check(a);
    b = SIMD.int16x8.check(b);
    return SIMD.int16x8(a.s0 | b.s0, a.s1 | b.s1, a.s2 | b.s2, a.s3 | b.s3,
                        a.s4 | b.s4, a.s5 | b.s5, a.s6 | b.s6, a.s7 | b.s7);
  }
}

if (typeof SIMD.int16x8.xor === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {int16x8} b An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of a ^ b.
    */
  SIMD.int16x8.xor = function(a, b) {
    a = SIMD.int16x8.check(a);
    b = SIMD.int16x8.check(b);
    return SIMD.int16x8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3,
                        a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7);
  }
}

if (typeof SIMD.int16x8.not === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of ~t
    */
  SIMD.int16x8.not = function(t) {
    t = SIMD.int16x8.check(t);
    return SIMD.int16x8(~t.s0, ~t.s1, ~t.s2, ~t.s3,
                        ~t.s4, ~t.s5, ~t.s6, ~t.s7);
  }
}

if (typeof SIMD.int16x8.neg === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of -t
    */
  SIMD.int16x8.neg = function(t) {
    t = SIMD.int16x8.check(t);
    return SIMD.int16x8(-t.s0, -t.s1, -t.s2, -t.s3,
                        -t.s4, -t.s5, -t.s6, -t.s7);
  }
}

if (typeof SIMD.int16x8.add === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {int16x8} b An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of a + b.
    */
  SIMD.int16x8.add = function(a, b) {
    a = SIMD.int16x8.check(a);
    b = SIMD.int16x8.check(b);
    return SIMD.int16x8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3,
                        a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7);
  }
}

if (typeof SIMD.int16x8.sub === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {int16x8} b An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of a - b.
    */
  SIMD.int16x8.sub = function(a, b) {
    a = SIMD.int16x8.check(a);
    b = SIMD.int16x8.check(b);
    return SIMD.int16x8(a.s0 - b.s0, a.s1 - b.s1, a.s2 - b.s2, a.s3 - b.s3,
                        a.s4 - b.s4, a.s5 - b.s5, a.s6 - b.s6, a.s7 - b.s7);
  }
}

if (typeof SIMD.int16x8.mul === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {int16x8} b An instance of int16x8.
    * @return {int16x8} New instance of int16x8 with values of a * b.
    */
  SIMD.int16x8.mul = function(a, b) {
    a = SIMD.int16x8.check(a);
    b = SIMD.int16x8.check(b);
    return SIMD.int16x8(Math.imul(a.s0, b.s0), Math.imul(a.s1, b.s1),
                        Math.imul(a.s2, b.s2), Math.imul(a.s3, b.s3),
                        Math.imul(a.s4, b.s4), Math.imul(a.s5, b.s5),
                        Math.imul(a.s6, b.s6), Math.imul(a.s7, b.s7));
  }
}

if (typeof SIMD.int16x8.select === "undefined") {
  /**
    * @param {int16x8} t Selector mask. An instance of int16x8
    * @param {int16x8} trueValue Pick lane from here if corresponding
    * selector lane is 0xFFFF
    * @param {int16x8} falseValue Pick lane from here if corresponding
    * selector lane is 0x0
    * @return {int16x8} Mix of lanes from trueValue or falseValue as
    * indicated
    */
  SIMD.int16x8.select = function(t, trueValue, falseValue) {
    t = SIMD.int16x8.check(t);
    trueValue = SIMD.int16x8.check(trueValue);
    falseValue = SIMD.int16x8.check(falseValue);
    var tr = SIMD.int16x8.and(t, trueValue);
    var fr = SIMD.int16x8.and(SIMD.int16x8.not(t), falseValue);
    return SIMD.int16x8.or(tr, fr);
  }
}

if (typeof SIMD.int16x8.equal === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @param {int16x8} other An instance of int16x8.
    * @return {int16x8} 0xFFFF or 0x0 in each lane depending on
    * the result of t == other.
    */
  SIMD.int16x8.equal = function(t, other) {
    t = SIMD.int16x8.check(t);
    other = SIMD.int16x8.check(other);
    var cs0 = t.s0 == other.s0;
    var cs1 = t.s1 == other.s1;
    var cs2 = t.s2 == other.s2;
    var cs3 = t.s3 == other.s3;
    var cs4 = t.s4 == other.s4;
    var cs5 = t.s5 == other.s5;
    var cs6 = t.s6 == other.s6;
    var cs7 = t.s7 == other.s7;
    return SIMD.int16x8.bool(cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
  }
}

if (typeof SIMD.int16x8.greaterThan === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @param {int16x8} other An instance of int16x8.
    * @return {int16x8} 0xFFFF or 0x0 in each lane depending on
    * the result of t > other.
    */
  SIMD.int16x8.greaterThan = function(t, other) {
    t = SIMD.int16x8.check(t);
    other = SIMD.int16x8.check(other);
    var cs0 = t.s0 > other.s0;
    var cs1 = t.s1 > other.s1;
    var cs2 = t.s2 > other.s2;
    var cs3 = t.s3 > other.s3;
    var cs4 = t.s4 > other.s4;
    var cs5 = t.s5 > other.s5;
    var cs6 = t.s6 > other.s6;
    var cs7 = t.s7 > other.s7;
    return SIMD.int16x8.bool(cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
  }
}

if (typeof SIMD.int16x8.lessThan === "undefined") {
  /**
    * @param {int16x8} t An instance of int16x8.
    * @param {int16x8} other An instance of int16x8.
    * @return {int16x8} 0xFFFF or 0x0 in each lane depending on
    * the result of t < other.
    */
  SIMD.int16x8.lessThan = function(t, other) {
    t = SIMD.int16x8.check(t);
    other = SIMD.int16x8.check(other);
    var cs0 = t.s0 < other.s0;
    var cs1 = t.s1 < other.s1;
    var cs2 = t.s2 < other.s2;
    var cs3 = t.s3 < other.s3;
    var cs4 = t.s4 < other.s4;
    var cs5 = t.s5 < other.s5;
    var cs6 = t.s6 < other.s6;
    var cs7 = t.s7 < other.s7;
    return SIMD.int16x8.bool(cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7);
  }
}

if (typeof SIMD.int16x8.shiftLeftByScalar === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {integer} bits Bit count to shift by.
    * @return {int16x8} lanes in a shifted by bits.
    */
  SIMD.int16x8.shiftLeftByScalar = function(a, bits) {
    a = SIMD.int16x8.check(a);
    var s0 = a.s0 << bits;
    var s1 = a.s1 << bits;
    var s2 = a.s2 << bits;
    var s3 = a.s3 << bits;
    var s4 = a.s4 << bits;
    var s5 = a.s5 << bits;
    var s6 = a.s6 << bits;
    var s7 = a.s7 << bits;
    return SIMD.int16x8(s0, s1, s2, s3, s4, s5, s6, s7);
  }
}

if (typeof SIMD.int16x8.shiftRightLogicalByScalar === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {integer} bits Bit count to shift by.
    * @return {int16x8} lanes in a shifted by bits.
    */
  SIMD.int16x8.shiftRightLogicalByScalar = function(a, bits) {
    a = SIMD.int16x8.check(a);
    var s0 = (a.s0 & 0xffff) >>> bits;
    var s1 = (a.s1 & 0xffff) >>> bits;
    var s2 = (a.s2 & 0xffff) >>> bits;
    var s3 = (a.s3 & 0xffff) >>> bits;
    var s4 = (a.s4 & 0xffff) >>> bits;
    var s5 = (a.s5 & 0xffff) >>> bits;
    var s6 = (a.s6 & 0xffff) >>> bits;
    var s7 = (a.s7 & 0xffff) >>> bits;
    return SIMD.int16x8(s0, s1, s2, s3, s4, s5, s6, s7);
  }
}

if (typeof SIMD.int16x8.shiftRightArithmeticByScalar === "undefined") {
  /**
    * @param {int16x8} a An instance of int16x8.
    * @param {integer} bits Bit count to shift by.
    * @return {int16x8} lanes in a shifted by bits.
    */
  SIMD.int16x8.shiftRightArithmeticByScalar = function(a, bits) {
    a = SIMD.int16x8.check(a);
    var s0 = a.s0 >> bits;
    var s1 = a.s1 >> bits;
    var s2 = a.s2 >> bits;
    var s3 = a.s3 >> bits;
    var s4 = a.s4 >> bits;
    var s5 = a.s5 >> bits;
    var s6 = a.s6 >> bits;
    var s7 = a.s7 >> bits;
    return SIMD.int16x8(s0, s1, s2, s3, s4, s5, s6, s7);
  }
}

if (typeof SIMD.int16x8.load === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {int16x8} New instance of int16x8.
    */
  SIMD.int16x8.load = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var i16temp = _SIMD_PRIVATE._i16x8;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? i16temp :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    return SIMD.int16x8(i16temp[0], i16temp[1], i16temp[2], i16temp[3],
                        i16temp[4], i16temp[5], i16temp[6], i16temp[7]);
  }
}

if (typeof SIMD.int16x8.store === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {int16x8} value An instance of int16x8.
    * @return {void}
    */
  SIMD.int16x8.store = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.int16x8.check(value);
    _SIMD_PRIVATE._i16x8[0] = value.s0;
    _SIMD_PRIVATE._i16x8[1] = value.s1;
    _SIMD_PRIVATE._i16x8[2] = value.s2;
    _SIMD_PRIVATE._i16x8[3] = value.s3;
    _SIMD_PRIVATE._i16x8[4] = value.s4;
    _SIMD_PRIVATE._i16x8[5] = value.s5;
    _SIMD_PRIVATE._i16x8[6] = value.s6;
    _SIMD_PRIVATE._i16x8[7] = value.s7;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof SIMD.int8x16.and === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {int8x16} b An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of a & b.
    */
  SIMD.int8x16.and = function(a, b) {
    a = SIMD.int8x16.check(a);
    b = SIMD.int8x16.check(b);
    return SIMD.int8x16(a.s0 & b.s0, a.s1 & b.s1, a.s2 & b.s2, a.s3 & b.s3,
                        a.s4 & b.s4, a.s5 & b.s5, a.s6 & b.s6, a.s7 & b.s7,
                        a.s8 & b.s8, a.s9 & b.s9, a.s10 & b.s10, a.s11 & b.s11,
                        a.s12 & b.s12, a.s13 & b.s13, a.s14 & b.s14, a.s15 & b.s15);
  }
}

if (typeof SIMD.int8x16.or === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {int8x16} b An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of a | b.
    */
  SIMD.int8x16.or = function(a, b) {
    a = SIMD.int8x16.check(a);
    b = SIMD.int8x16.check(b);
    return SIMD.int8x16(a.s0 | b.s0, a.s1 | b.s1, a.s2 | b.s2, a.s3 | b.s3,
                        a.s4 | b.s4, a.s5 | b.s5, a.s6 | b.s6, a.s7 | b.s7,
                        a.s8 | b.s8, a.s9 | b.s9, a.s10 | b.s10, a.s11 | b.s11,
                        a.s12 | b.s12, a.s13 | b.s13, a.s14 | b.s14, a.s15 | b.s15);
  }
}

if (typeof SIMD.int8x16.xor === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {int8x16} b An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of a ^ b.
    */
  SIMD.int8x16.xor = function(a, b) {
    a = SIMD.int8x16.check(a);
    b = SIMD.int8x16.check(b);
    return SIMD.int8x16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3,
                        a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
                        a.s8 ^ b.s8, a.s9 ^ b.s9, a.s10 ^ b.s10, a.s11 ^ b.s11,
                        a.s12 ^ b.s12, a.s13 ^ b.s13, a.s14 ^ b.s14, a.s15 ^ b.s15);
  }
}

if (typeof SIMD.int8x16.not === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of ~t
    */
  SIMD.int8x16.not = function(t) {
    t = SIMD.int8x16.check(t);
    return SIMD.int8x16(~t.s0, ~t.s1, ~t.s2, ~t.s3,
                        ~t.s4, ~t.s5, ~t.s6, ~t.s7,
                        ~t.s8, ~t.s9, ~t.s10, ~t.s11,
                        ~t.s12, ~t.s13, ~t.s14, ~t.s15);
  }
}

if (typeof SIMD.int8x16.neg === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of -t
    */
  SIMD.int8x16.neg = function(t) {
    t = SIMD.int8x16.check(t);
    return SIMD.int8x16(-t.s0, -t.s1, -t.s2, -t.s3,
                        -t.s4, -t.s5, -t.s6, -t.s7,
                        -t.s8, -t.s9, -t.s10, -t.s11,
                        -t.s12, -t.s13, -t.s14, -t.s15);
  }
}

if (typeof SIMD.int8x16.add === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {int8x16} b An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of a + b.
    */
  SIMD.int8x16.add = function(a, b) {
    a = SIMD.int8x16.check(a);
    b = SIMD.int8x16.check(b);
    return SIMD.int8x16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3,
                        a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
                        a.s8 + b.s8, a.s9 + b.s9, a.s10 + b.s10, a.s11 + b.s11,
                        a.s12 + b.s12, a.s13 + b.s13, a.s14 + b.s14, a.s15 + b.s15);
  }
}

if (typeof SIMD.int8x16.sub === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {int8x16} b An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of a - b.
    */
  SIMD.int8x16.sub = function(a, b) {
    a = SIMD.int8x16.check(a);
    b = SIMD.int8x16.check(b);
    return SIMD.int8x16(a.s0 - b.s0, a.s1 - b.s1, a.s2 - b.s2, a.s3 - b.s3,
                        a.s4 - b.s4, a.s5 - b.s5, a.s6 - b.s6, a.s7 - b.s7,
                        a.s8 - b.s8, a.s9 - b.s9, a.s10 - b.s10, a.s11 - b.s11,
                        a.s12 - b.s12, a.s13 - b.s13, a.s14 - b.s14, a.s15 - b.s15);
  }
}

if (typeof SIMD.int8x16.mul === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {int8x16} b An instance of int8x16.
    * @return {int8x16} New instance of int8x16 with values of a * b.
    */
  SIMD.int8x16.mul = function(a, b) {
    a = SIMD.int8x16.check(a);
    b = SIMD.int8x16.check(b);
    return SIMD.int8x16(Math.imul(a.s0, b.s0), Math.imul(a.s1, b.s1),
                        Math.imul(a.s2, b.s2), Math.imul(a.s3, b.s3),
                        Math.imul(a.s4, b.s4), Math.imul(a.s5, b.s5),
                        Math.imul(a.s6, b.s6), Math.imul(a.s7, b.s7),
                        Math.imul(a.s8, b.s8), Math.imul(a.s9, b.s9),
                        Math.imul(a.s10, b.s10), Math.imul(a.s11, b.s11),
                        Math.imul(a.s12, b.s12), Math.imul(a.s13, b.s13),
                        Math.imul(a.s14, b.s14), Math.imul(a.s15, b.s15));
  }
}

if (typeof SIMD.int8x16.select === "undefined") {
  /**
    * @param {int8x16} t Selector mask. An instance of int8x16
    * @param {int8x16} trueValue Pick lane from here if corresponding
    * selector lane is 0xFF
    * @param {int8x16} falseValue Pick lane from here if corresponding
    * selector lane is 0x0
    * @return {int8x16} Mix of lanes from trueValue or falseValue as
    * indicated
    */
  SIMD.int8x16.select = function(t, trueValue, falseValue) {
    t = SIMD.int8x16.check(t);
    trueValue = SIMD.int8x16.check(trueValue);
    falseValue = SIMD.int8x16.check(falseValue);
    var tr = SIMD.int8x16.and(t, trueValue);
    var fr = SIMD.int8x16.and(SIMD.int8x16.not(t), falseValue);
    return SIMD.int8x16.or(tr, fr);
  }
}

if (typeof SIMD.int8x16.equal === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @param {int8x16} other An instance of int8x16.
    * @return {int8x16} 0xFF or 0x0 in each lane depending on
    * the result of t == other.
    */
  SIMD.int8x16.equal = function(t, other) {
    t = SIMD.int8x16.check(t);
    other = SIMD.int8x16.check(other);
    var cs0 = t.s0 == other.s0;
    var cs1 = t.s1 == other.s1;
    var cs2 = t.s2 == other.s2;
    var cs3 = t.s3 == other.s3;
    var cs4 = t.s4 == other.s4;
    var cs5 = t.s5 == other.s5;
    var cs6 = t.s6 == other.s6;
    var cs7 = t.s7 == other.s7;
    var cs8 = t.s8 == other.s8;
    var cs9 = t.s9 == other.s9;
    var cs10 = t.s10 == other.s10;
    var cs11 = t.s11 == other.s11;
    var cs12 = t.s12 == other.s12;
    var cs13 = t.s13 == other.s13;
    var cs14 = t.s14 == other.s14;
    var cs15 = t.s15 == other.s15;
    return SIMD.int8x16.bool(cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7,
                             cs8, cs9, cs10, cs11, cs12, cs13, cs14, cs15);
  }
}

if (typeof SIMD.int8x16.greaterThan === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @param {int8x16} other An instance of int8x16.
    * @return {int8x16} 0xFF or 0x0 in each lane depending on
    * the result of t > other.
    */
  SIMD.int8x16.greaterThan = function(t, other) {
    t = SIMD.int8x16.check(t);
    other = SIMD.int8x16.check(other);
    var cs0 = t.s0 > other.s0;
    var cs1 = t.s1 > other.s1;
    var cs2 = t.s2 > other.s2;
    var cs3 = t.s3 > other.s3;
    var cs4 = t.s4 > other.s4;
    var cs5 = t.s5 > other.s5;
    var cs6 = t.s6 > other.s6;
    var cs7 = t.s7 > other.s7;
    var cs8 = t.s8 > other.s8;
    var cs9 = t.s9 > other.s9;
    var cs10 = t.s10 > other.s10;
    var cs11 = t.s11 > other.s11;
    var cs12 = t.s12 > other.s12;
    var cs13 = t.s13 > other.s13;
    var cs14 = t.s14 > other.s14;
    var cs15 = t.s15 > other.s15;
    return SIMD.int8x16.bool(cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7,
                             cs8, cs9, cs10, cs11, cs12, cs13, cs14, cs15);
  }
}

if (typeof SIMD.int8x16.lessThan === "undefined") {
  /**
    * @param {int8x16} t An instance of int8x16.
    * @param {int8x16} other An instance of int8x16.
    * @return {int8x16} 0xFF or 0x0 in each lane depending on
    * the result of t < other.
    */
  SIMD.int8x16.lessThan = function(t, other) {
    t = SIMD.int8x16.check(t);
    other = SIMD.int8x16.check(other);
    var cs0 = t.s0 < other.s0;
    var cs1 = t.s1 < other.s1;
    var cs2 = t.s2 < other.s2;
    var cs3 = t.s3 < other.s3;
    var cs4 = t.s4 < other.s4;
    var cs5 = t.s5 < other.s5;
    var cs6 = t.s6 < other.s6;
    var cs7 = t.s7 < other.s7;
    var cs8 = t.s8 < other.s8;
    var cs9 = t.s9 < other.s9;
    var cs10 = t.s10 < other.s10;
    var cs11 = t.s11 < other.s11;
    var cs12 = t.s12 < other.s12;
    var cs13 = t.s13 < other.s13;
    var cs14 = t.s14 < other.s14;
    var cs15 = t.s15 < other.s15;
    return SIMD.int8x16.bool(cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7,
                             cs8, cs9, cs10, cs11, cs12, cs13, cs14, cs15);
  }
}

if (typeof SIMD.int8x16.shiftLeftByScalar === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {integer} bits Bit count to shift by.
    * @return {int8x16} lanes in a shifted by bits.
    */
  SIMD.int8x16.shiftLeftByScalar = function(a, bits) {
    a = SIMD.int8x16.check(a);
    var s0 = a.s0 << bits;
    var s1 = a.s1 << bits;
    var s2 = a.s2 << bits;
    var s3 = a.s3 << bits;
    var s4 = a.s4 << bits;
    var s5 = a.s5 << bits;
    var s6 = a.s6 << bits;
    var s7 = a.s7 << bits;
    var s8 = a.s8 << bits;
    var s9 = a.s9 << bits;
    var s10 = a.s10 << bits;
    var s11 = a.s11 << bits;
    var s12 = a.s12 << bits;
    var s13 = a.s13 << bits;
    var s14 = a.s14 << bits;
    var s15 = a.s15 << bits;
    return SIMD.int8x16(s0, s1, s2, s3, s4, s5, s6, s7,
                        s8, s9, s10, s11, s12, s13, s14, s15);
  }
}

if (typeof SIMD.int8x16.shiftRightLogicalByScalar === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {integer} bits Bit count to shift by.
    * @return {int8x16} lanes in a shifted by bits.
    */
  SIMD.int8x16.shiftRightLogicalByScalar = function(a, bits) {
    a = SIMD.int8x16.check(a);
    var s0 = (a.s0 & 0xff) >>> bits;
    var s1 = (a.s1 & 0xff) >>> bits;
    var s2 = (a.s2 & 0xff) >>> bits;
    var s3 = (a.s3 & 0xff) >>> bits;
    var s4 = (a.s4 & 0xff) >>> bits;
    var s5 = (a.s5 & 0xff) >>> bits;
    var s6 = (a.s6 & 0xff) >>> bits;
    var s7 = (a.s7 & 0xff) >>> bits;
    var s8 = (a.s8 & 0xff) >>> bits;
    var s9 = (a.s9 & 0xff) >>> bits;
    var s10 = (a.s10 & 0xff) >>> bits;
    var s11 = (a.s11 & 0xff) >>> bits;
    var s12 = (a.s12 & 0xff) >>> bits;
    var s13 = (a.s13 & 0xff) >>> bits;
    var s14 = (a.s14 & 0xff) >>> bits;
    var s15 = (a.s15 & 0xff) >>> bits;
    return SIMD.int8x16(s0, s1, s2, s3, s4, s5, s6, s7,
                        s8, s9, s10, s11, s12, s13, s14, s15);
  }
}

if (typeof SIMD.int8x16.shiftRightArithmeticByScalar === "undefined") {
  /**
    * @param {int8x16} a An instance of int8x16.
    * @param {integer} bits Bit count to shift by.
    * @return {int8x16} lanes in a shifted by bits.
    */
  SIMD.int8x16.shiftRightArithmeticByScalar = function(a, bits) {
    a = SIMD.int8x16.check(a);
    var s0 = a.s0 >> bits;
    var s1 = a.s1 >> bits;
    var s2 = a.s2 >> bits;
    var s3 = a.s3 >> bits;
    var s4 = a.s4 >> bits;
    var s5 = a.s5 >> bits;
    var s6 = a.s6 >> bits;
    var s7 = a.s7 >> bits;
    var s8 = a.s8 >> bits;
    var s9 = a.s9 >> bits;
    var s10 = a.s10 >> bits;
    var s11 = a.s11 >> bits;
    var s12 = a.s12 >> bits;
    var s13 = a.s13 >> bits;
    var s14 = a.s14 >> bits;
    var s15 = a.s15 >> bits;
    return SIMD.int8x16(s0, s1, s2, s3, s4, s5, s6, s7,
                        s8, s9, s10, s11, s12, s13, s14, s15);
  }
}

if (typeof SIMD.int8x16.load === "undefined") {
  /**
    * @param {Typed array} buffer An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @return {int8x16} New instance of int8x16.
    */
  SIMD.int8x16.load = function(tarray, index) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    var i8temp = _SIMD_PRIVATE._i8x16;
    var array = bpe == 1 ? i8temp :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      array[i] = tarray[index + i];
    return SIMD.int8x16(i8temp[0], i8temp[1], i8temp[2], i8temp[3],
                        i8temp[4], i8temp[5], i8temp[6], i8temp[7],
                        i8temp[8], i8temp[9], i8temp[10], i8temp[11],
                        i8temp[12], i8temp[13], i8temp[14], i8temp[15]);
  }
}

if (typeof SIMD.int8x16.store === "undefined") {
  /**
    * @param {Typed array} tarray An instance of a typed array.
    * @param {Number} index An instance of Number.
    * @param {int8x16} value An instance of int8x16.
    * @return {void}
    */
  SIMD.int8x16.store = function(tarray, index, value) {
    if (!_SIMD_PRIVATE.isTypedArray(tarray))
      throw new TypeError("The 1st argument must be a typed array.");
    if (!_SIMD_PRIVATE.isNumber(index))
      throw new TypeError("The 2nd argument must be a Number.");
    var bpe = tarray.BYTES_PER_ELEMENT;
    if (index < 0 || (index * bpe + 16) > tarray.byteLength)
      throw new RangeError("The value of index is invalid.");
    value = SIMD.int8x16.check(value);
    _SIMD_PRIVATE._i8x16[0] = value.s0;
    _SIMD_PRIVATE._i8x16[1] = value.s1;
    _SIMD_PRIVATE._i8x16[2] = value.s2;
    _SIMD_PRIVATE._i8x16[3] = value.s3;
    _SIMD_PRIVATE._i8x16[4] = value.s4;
    _SIMD_PRIVATE._i8x16[5] = value.s5;
    _SIMD_PRIVATE._i8x16[6] = value.s6;
    _SIMD_PRIVATE._i8x16[7] = value.s7;
    _SIMD_PRIVATE._i8x16[8] = value.s8;
    _SIMD_PRIVATE._i8x16[9] = value.s9;
    _SIMD_PRIVATE._i8x16[10] = value.s10;
    _SIMD_PRIVATE._i8x16[11] = value.s11;
    _SIMD_PRIVATE._i8x16[12] = value.s12;
    _SIMD_PRIVATE._i8x16[13] = value.s13;
    _SIMD_PRIVATE._i8x16[14] = value.s14;
    _SIMD_PRIVATE._i8x16[15] = value.s15;
    var array = bpe == 1 ? _SIMD_PRIVATE._i8x16 :
                bpe == 2 ? _SIMD_PRIVATE._i16x8 :
                bpe == 4 ? (tarray instanceof Float32Array ? _SIMD_PRIVATE._f32x4 : _SIMD_PRIVATE._i32x4) :
                _SIMD_PRIVATE._f64x2;
    var n = 16 / bpe;
    for (var i = 0; i < n; ++i)
      tarray[index + i] = array[i];
  }
}

if (typeof Float32x4Array === "undefined") {
  Float32x4Array = function(a, b, c) {
    if (_SIMD_PRIVATE.isNumber(a)) {
      this.storage_ = new Float32Array(a*4);
      this.length_ = a;
      this.byteOffset_ = 0;
      return;
    } else if (_SIMD_PRIVATE.isTypedArray(a)) {
      if (!(a instanceof Float32x4Array)) {
        throw "Copying typed array of non-Float32x4Array is unimplemented.";
      }
      this.storage_ = new Float32Array(a.length * 4);
      this.length_ = a.length;
      this.byteOffset_ = 0;
      // Copy floats.
      for (var i = 0; i < a.length*4; i++) {
        this.storage_[i] = a.storage_[i];
      }
    } else if (_SIMD_PRIVATE.isArrayBuffer(a)) {
      if ((b != undefined) && (b % Float32x4Array.BYTES_PER_ELEMENT) != 0) {
        throw "byteOffset must be a multiple of 16.";
      }
      if (c != undefined) {
        c *= 4;
        this.storage_ = new Float32Array(a, b, c);
      }
      else {
        // Note: new Float32Array(a, b) is NOT equivalent to new Float32Array(a, b, undefined)
        this.storage_ = new Float32Array(a, b);
      }
      this.length_ = this.storage_.length / 4;
      this.byteOffset_ = b != undefined ? b : 0;
    } else {
      throw "Unknown type of first argument.";
    }
  }

  Object.defineProperty(Float32x4Array.prototype, 'length', {
    get: function() { return this.length_; }
  });

  Object.defineProperty(Float32x4Array.prototype, 'byteLength', {
    get: function() { return this.length_ * Float32x4Array.BYTES_PER_ELEMENT; }
  });

  Object.defineProperty(Float32x4Array, 'BYTES_PER_ELEMENT', {
    get: function() { return 16; }
  });

  Object.defineProperty(Float32x4Array.prototype, 'BYTES_PER_ELEMENT', {
    get: function() { return 16; }
  });

  Object.defineProperty(Float32x4Array.prototype, 'byteOffset', {
    get: function() { return this.byteOffset_; }
  });

  Object.defineProperty(Float32x4Array.prototype, 'buffer', {
    get: function() { return this.storage_.buffer; }
  });

  Float32x4Array.prototype.getAt = function(i) {
    if (i < 0) {
      throw "Index must be >= 0.";
    }
    if (i >= this.length) {
      throw "Index out of bounds.";
    }
    var x = this.storage_[i*4+0];
    var y = this.storage_[i*4+1];
    var z = this.storage_[i*4+2];
    var w = this.storage_[i*4+3];
    return SIMD.float32x4(x, y, z, w);
  }

  Float32x4Array.prototype.setAt = function(i, v) {
    if (i < 0) {
      throw "Index must be >= 0.";
    }
    if (i >= this.length) {
      throw "Index out of bounds.";
    }
    if (!(v instanceof SIMD.float32x4)) {
      throw "Value is not a float32x4.";
    }
    this.storage_[i*4+0] = v.x;
    this.storage_[i*4+1] = v.y;
    this.storage_[i*4+2] = v.z;
    this.storage_[i*4+3] = v.w;
  }
}

if (typeof Int32x4Array === "undefined") {
  Int32x4Array = function(a, b, c) {
    if (_SIMD_PRIVATE.isNumber(a)) {
      this.storage_ = new Int32Array(a*4);
      this.length_ = a;
      this.byteOffset_ = 0;
      return;
    } else if (_SIMD_PRIVATE.isTypedArray(a)) {
      if (!(a instanceof Int32x4Array)) {
        throw "Copying typed array of non-Int32x4Array is unimplemented.";
      }
      this.storage_ = new Int32Array(a.length * 4);
      this.length_ = a.length;
      this.byteOffset_ = 0;
      // Copy ints.
      for (var i = 0; i < a.length*4; i++) {
        this.storage_[i] = a.storage_[i];
      }
    } else if (_SIMD_PRIVATE.isArrayBuffer(a)) {
      if ((b != undefined) && (b % Int32x4Array.BYTES_PER_ELEMENT) != 0) {
        throw "byteOffset must be a multiple of 16.";
      }
      if (c != undefined) {
        c *= 4;
        this.storage_ = new Int32Array(a, b, c);
      }
      else {
        // Note: new Int32Array(a, b) is NOT equivalent to new Int32Array(a, b, undefined)
        this.storage_ = new Int32Array(a, b);
      }
      this.length_ = this.storage_.length / 4;
      this.byteOffset_ = b != undefined ? b : 0;
    } else {
      throw "Unknown type of first argument.";
    }
  }

  Object.defineProperty(Int32x4Array.prototype, 'length', {
    get: function() { return this.length_; }
  });

  Object.defineProperty(Int32x4Array.prototype, 'byteLength', {
    get: function() { return this.length_ * Int32x4Array.BYTES_PER_ELEMENT; }
  });

  Object.defineProperty(Int32x4Array, 'BYTES_PER_ELEMENT', {
    get: function() { return 16; }
  });

  Object.defineProperty(Int32x4Array.prototype, 'BYTES_PER_ELEMENT', {
    get: function() { return 16; }
  });

  Object.defineProperty(Int32x4Array.prototype, 'byteOffset', {
    get: function() { return this.byteOffset_; }
  });

  Object.defineProperty(Int32x4Array.prototype, 'buffer', {
    get: function() { return this.storage_.buffer; }
  });

  Int32x4Array.prototype.getAt = function(i) {
    if (i < 0) {
      throw "Index must be >= 0.";
    }
    if (i >= this.length) {
      throw "Index out of bounds.";
    }
    var x = this.storage_[i*4+0];
    var y = this.storage_[i*4+1];
    var z = this.storage_[i*4+2];
    var w = this.storage_[i*4+3];
    return SIMD.int32x4(x, y, z, w);
  }

  Int32x4Array.prototype.setAt = function(i, v) {
    if (i < 0) {
      throw "Index must be >= 0.";
    }
    if (i >= this.length) {
      throw "Index out of bounds.";
    }
    if (!(v instanceof SIMD.int32x4)) {
      throw "Value is not a int32x4.";
    }
    this.storage_[i*4+0] = v.x;
    this.storage_[i*4+1] = v.y;
    this.storage_[i*4+2] = v.z;
    this.storage_[i*4+3] = v.w;
  }

  _SIMD_PRIVATE.isDataView = function(v) {
    return v instanceof DataView;
  }

  DataView.prototype.getFloat32x4 = function(byteOffset, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    return SIMD.float32x4(this.getFloat32(byteOffset, littleEndian),
                          this.getFloat32(byteOffset + 4, littleEndian),
                          this.getFloat32(byteOffset + 8, littleEndian),
                          this.getFloat32(byteOffset + 12, littleEndian));
  }

  DataView.prototype.getFloat64x2 = function(byteOffset, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    return SIMD.float64x2(this.getFloat64(byteOffset, littleEndian),
                          this.getFloat64(byteOffset + 8, littleEndian));
  }

  DataView.prototype.getInt32x4 = function(byteOffset, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    return SIMD.int32x4(this.getInt32(byteOffset, littleEndian),
                        this.getInt32(byteOffset + 4, littleEndian),
                        this.getInt32(byteOffset + 8, littleEndian),
                        this.getInt32(byteOffset + 12, littleEndian));
  }

  DataView.prototype.getInt16x8 = function(byteOffset, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    return SIMD.int16x8(this.getInt16(byteOffset, littleEndian),
                        this.getInt16(byteOffset + 2, littleEndian),
                        this.getInt16(byteOffset + 4, littleEndian),
                        this.getInt16(byteOffset + 6, littleEndian),
                        this.getInt16(byteOffset + 8, littleEndian),
                        this.getInt16(byteOffset + 10, littleEndian),
                        this.getInt16(byteOffset + 12, littleEndian),
                        this.getInt16(byteOffset + 14, littleEndian));
  }

  DataView.prototype.getInt8x16 = function(byteOffset, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    return SIMD.int8x16(this.getInt8(byteOffset, littleEndian),
                        this.getInt8(byteOffset + 1, littleEndian),
                        this.getInt8(byteOffset + 2, littleEndian),
                        this.getInt8(byteOffset + 3, littleEndian),
                        this.getInt8(byteOffset + 4, littleEndian),
                        this.getInt8(byteOffset + 5, littleEndian),
                        this.getInt8(byteOffset + 6, littleEndian),
                        this.getInt8(byteOffset + 7, littleEndian),
                        this.getInt8(byteOffset + 8, littleEndian),
                        this.getInt8(byteOffset + 9, littleEndian),
                        this.getInt8(byteOffset + 10, littleEndian),
                        this.getInt8(byteOffset + 11, littleEndian),
                        this.getInt8(byteOffset + 12, littleEndian),
                        this.getInt8(byteOffset + 13, littleEndian),
                        this.getInt8(byteOffset + 14, littleEndian),
                        this.getInt8(byteOffset + 15, littleEndian));
  }

  DataView.prototype.setFloat32x4 = function(byteOffset, value, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    value = SIMD.float32x4.check(value);
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    this.setFloat32(byteOffset, value.x, littleEndian);
    this.setFloat32(byteOffset + 4, value.y, littleEndian);
    this.setFloat32(byteOffset + 8, value.z, littleEndian);
    this.setFloat32(byteOffset + 12, value.w, littleEndian);
  }

  DataView.prototype.setFloat64x2 = function(byteOffset, value, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    value = SIMD.float64x2.check(value);
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    this.setFloat64(byteOffset, value.x, littleEndian);
    this.setFloat64(byteOffset + 8, value.y, littleEndian);
  }

  DataView.prototype.setInt32x4 = function(byteOffset, value, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    value = SIMD.int32x4.check(value);
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    this.setInt32(byteOffset, value.x, littleEndian);
    this.setInt32(byteOffset + 4, value.y, littleEndian);
    this.setInt32(byteOffset + 8, value.z, littleEndian);
    this.setInt32(byteOffset + 12, value.w, littleEndian);
  }

  DataView.prototype.setInt16x8 = function(byteOffset, value, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    value = SIMD.int16x8.check(value);
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    this.setInt16(byteOffset, value.s0, littleEndian);
    this.setInt16(byteOffset + 2, value.s1, littleEndian);
    this.setInt16(byteOffset + 4, value.s2, littleEndian);
    this.setInt16(byteOffset + 6, value.s3, littleEndian);
    this.setInt16(byteOffset + 8, value.s4, littleEndian);
    this.setInt16(byteOffset + 10, value.s5, littleEndian);
    this.setInt16(byteOffset + 12, value.s6, littleEndian);
    this.setInt16(byteOffset + 14, value.s7, littleEndian);
  }

  DataView.prototype.setInt8x16 = function(byteOffset, value, littleEndian) {
    if (!_SIMD_PRIVATE.isDataView(this))
      throw new TypeError("This is not a DataView.");
    if (byteOffset < 0 || (byteOffset + 16) > this.buffer.byteLength)
      throw new RangeError("The value of byteOffset is invalid.");
    value = SIMD.int8x16.check(value);
    if (typeof littleEndian === 'undefined')
      littleEndian = false;
    this.setInt8(byteOffset, value.s0, littleEndian);
    this.setInt8(byteOffset + 1, value.s1, littleEndian);
    this.setInt8(byteOffset + 2, value.s2, littleEndian);
    this.setInt8(byteOffset + 3, value.s3, littleEndian);
    this.setInt8(byteOffset + 4, value.s4, littleEndian);
    this.setInt8(byteOffset + 5, value.s5, littleEndian);
    this.setInt8(byteOffset + 6, value.s6, littleEndian);
    this.setInt8(byteOffset + 7, value.s7, littleEndian);
    this.setInt8(byteOffset + 8, value.s8, littleEndian);
    this.setInt8(byteOffset + 9, value.s9, littleEndian);
    this.setInt8(byteOffset + 10, value.s10, littleEndian);
    this.setInt8(byteOffset + 11, value.s11, littleEndian);
    this.setInt8(byteOffset + 12, value.s12, littleEndian);
    this.setInt8(byteOffset + 13, value.s13, littleEndian);
    this.setInt8(byteOffset + 14, value.s14, littleEndian);
    this.setInt8(byteOffset + 15, value.s15, littleEndian);
  }
}

if (typeof SIMD === 'undefined') {
    quit(0);
}

var assertEq = assertEq || function(a, b) { if (a !== b) throw new Error("assertion error: obtained " + a + ", expected " + b); };

// global variables
const MAX_ITERATIONS = 100;
const DRAW_ITERATIONS = 20;

const CANVAS_WIDTH = 100;
const CANVAS_HEIGHT = 100;

const LIMIT_SHOW = 20 * 20 * 4;

// Asm.js module buffer.
var buffer = new ArrayBuffer(16 * 1024 * 1024);
var view = new Uint8Array(buffer);

function moduleCode(global, ffi, buffer) {
  var b8 = new global.Uint8Array(buffer);
  var toF = global.Math.fround;
  var i4 = global.SIMD.int32x4;
  var f4 = global.SIMD.float32x4;
  var i4add = i4.add;
  var i4and = i4.and;
  var i4check = i4.check;
  var f4add = f4.add;
  var f4sub = f4.sub;
  var f4mul = f4.mul;
  var f4lessThanOrEqual = f4.lessThanOrEqual;
  var f4splat = f4.splat;
  var imul = global.Math.imul;
  const one4 = i4(1,1,1,1), two4 = f4(2,2,2,2), four4 = f4(4,4,4,4);

  const mk0 = 0x007fffff;

  function declareHeapLength() {
    b8[0x00ffffff] = 0;
  }

  function mapColorAndSetPixel (x, y, width, value, max_iterations) {
    x = x | 0;
    y = y | 0;
    width = width | 0;
    value = value | 0;
    max_iterations = max_iterations | 0;

    var rgb = 0, r = 0, g = 0, b = 0, index = 0;

    index = (((imul((width >>> 0), (y >>> 0)) + x) | 0) * 4) | 0;
    if ((value | 0) == (max_iterations | 0)) {
      r = 0;
      g = 0;
      b = 0;
    } else {
      rgb = ~~toF(toF(toF(toF(value >>> 0) * toF(0xffff)) / toF(max_iterations >>> 0)) * toF(0xff));
      r = rgb & 0xff;
      g = (rgb >>> 8) & 0xff;
      b = (rgb >>> 16) & 0xff;
    }
    b8[(index & mk0) >> 0] = r;
    b8[(index & mk0) + 1 >> 0] = g;
    b8[(index & mk0) + 2 >> 0] = b;
    b8[(index & mk0) + 3 >> 0] = 255;
  }

  function mandelPixelX4 (xf, yf, yd, max_iterations) {
    xf = toF(xf);
    yf = toF(yf);
    yd = toF(yd);
    max_iterations = max_iterations | 0;
    var c_re4  = f4(0,0,0,0), c_im4  = f4(0,0,0,0);
    var z_re4  = f4(0,0,0,0), z_im4  = f4(0,0,0,0);
    var count4 = i4(0,0,0,0);
    var z_re24 = f4(0,0,0,0), z_im24 = f4(0,0,0,0);
    var new_re4 = f4(0,0,0,0), new_im4 = f4(0,0,0,0);
    var i = 0;
    var mi4 = i4(0,0,0,0);

    c_re4 = f4splat(xf);
    c_im4 = f4(yf, toF(yd + yf), toF(yd + toF(yd + yf)), toF(yd + toF(yd + toF(yd + yf))));

    z_re4  = c_re4;
    z_im4  = c_im4;

    for (i = 0; (i | 0) < (max_iterations | 0); i = (i + 1) | 0) {
      z_re24 = f4mul(z_re4, z_re4);
      z_im24 = f4mul(z_im4, z_im4);

      mi4 = f4lessThanOrEqual(f4add(z_re24, z_im24), four4);
      // If all 4 values are greater than 4.0, there's no reason to continue.
      if ((mi4.signMask | 0) == 0x00)
        break;

      new_re4 = f4sub(z_re24, z_im24);
      new_im4 = f4mul(f4mul(two4, z_re4), z_im4);
      z_re4   = f4add(c_re4, new_re4);
      z_im4   = f4add(c_im4, new_im4);
      count4  = i4add(count4, i4and(mi4, one4));
    }
    return i4check(count4);
  }

  function mandelColumnX4 (x, width, height, xf, yf, yd, max_iterations) {
    x = x | 0;
    width = width | 0;
    height = height | 0;
    xf = toF(xf);
    yf = toF(yf);
    yd = toF(yd);
    max_iterations = max_iterations | 0;

    var y = 0;
    var ydx4 = toF(0);
    var m4 = i4(0,0,0,0);

    ydx4 = toF(yd * toF(4));
    for (y = 0; (y | 0) < (height | 0); y = (y + 4) | 0) {
      m4   = i4check(mandelPixelX4(toF(xf), toF(yf), toF(yd), max_iterations));
      mapColorAndSetPixel(x | 0, y | 0,   width, m4.x, max_iterations);
      mapColorAndSetPixel(x | 0, (y + 1) | 0, width, m4.y, max_iterations);
      mapColorAndSetPixel(x | 0, (y + 2) | 0, width, m4.z, max_iterations);
      mapColorAndSetPixel(x | 0, (y + 3) | 0, width, m4.w, max_iterations);
      yf = toF(yf + ydx4);
    }
  }

  function mandel (width, height, xc, yc, scale, max_iterations) {
    width = width | 0;
    height = height | 0;
    xc = toF(xc);
    yc = toF(yc);
    scale = toF(scale);
    max_iterations = max_iterations | 0;

    var x0 = toF(0), y0 = toF(0);
    var xd = toF(0), yd = toF(0);
    var xf = toF(0);
    var x = 0;

    x0 = toF(xc - toF(scale * toF(1.5)));
    y0 = toF(yc - scale);
    xd = toF(toF(scale * toF(3)) / toF(width >>> 0));
    yd = toF(toF(scale * toF(2)) / toF(height >>> 0));
    xf = x0;

    for (x = 0; (x | 0) < (width | 0); x = (x + 1) | 0) {
      mandelColumnX4(x, width, height, xf, y0, yd, max_iterations);
      xf = toF(xf + xd);
    }
  }

  return mandel;
};

var FFI = {};

var mandelbro = moduleCode(this, FFI, buffer);

function animateMandelbrot () {
  var scale_start = 1.0;
  var scale_end   = 0.0005;
  var xc_start    = -0.5;
  var yc_start    = 0.0;
  var xc_end      = 0.0;
  var yc_end      = 0.75;
  var steps       = 200.0;
  var scale_step  = (scale_end - scale_start)/steps;
  var xc_step     = (xc_end - xc_start)/steps;
  var yc_step     = (yc_end - yc_start)/steps;
  var scale       = scale_start;
  var xc          = xc_start;
  var yc          = yc_start;
  var i           = 0;

  function draw1 () {
    mandelbro(CANVAS_WIDTH, CANVAS_HEIGHT, xc, yc, scale, MAX_ITERATIONS);
    if (scale < scale_end || scale > scale_start) {
      scale_step = -scale_step;
      xc_step = -xc_step;
      yc_step = -yc_step;
    }
    scale += scale_step;
    xc += xc_step;
    yc += yc_step;
    i++;
  }

  for (var j = DRAW_ITERATIONS; j --> 0;)
    draw1();
}

animateMandelbrot();

assertEq(view[0], 202, "0th value should be 202");
assertEq(view[1], 140, "1th value should be 140");
assertEq(view[2], 2, "2th value should be 2");
assertEq(view[3], 255, "3th value should be 255");
assertEq(view[4], 202, "4th value should be 202");
assertEq(view[5], 140, "5th value should be 140");
assertEq(view[6], 2, "6th value should be 2");
assertEq(view[7], 255, "7th value should be 255");
assertEq(view[8], 202, "8th value should be 202");
assertEq(view[9], 140, "9th value should be 140");
assertEq(view[10], 2, "10th value should be 2");
assertEq(view[11], 255, "11th value should be 255");
assertEq(view[12], 202, "12th value should be 202");
assertEq(view[13], 140, "13th value should be 140");
assertEq(view[14], 2, "14th value should be 2");
assertEq(view[15], 255, "15th value should be 255");
assertEq(view[16], 202, "16th value should be 202");
assertEq(view[17], 140, "17th value should be 140");
assertEq(view[18], 2, "18th value should be 2");
assertEq(view[19], 255, "19th value should be 255");
assertEq(view[20], 202, "20th value should be 202");
assertEq(view[21], 140, "21th value should be 140");
assertEq(view[22], 2, "22th value should be 2");
assertEq(view[23], 255, "23th value should be 255");
assertEq(view[24], 148, "24th value should be 148");
assertEq(view[25], 25, "25th value should be 25");
assertEq(view[26], 5, "26th value should be 5");
assertEq(view[27], 255, "27th value should be 255");
assertEq(view[28], 148, "28th value should be 148");
assertEq(view[29], 25, "29th value should be 25");
assertEq(view[30], 5, "30th value should be 5");
assertEq(view[31], 255, "31th value should be 255");
assertEq(view[32], 148, "32th value should be 148");
assertEq(view[33], 25, "33th value should be 25");
assertEq(view[34], 5, "34th value should be 5");
assertEq(view[35], 255, "35th value should be 255");
assertEq(view[36], 148, "36th value should be 148");
assertEq(view[37], 25, "37th value should be 25");
assertEq(view[38], 5, "38th value should be 5");
assertEq(view[39], 255, "39th value should be 255");
assertEq(view[40], 148, "40th value should be 148");
assertEq(view[41], 25, "41th value should be 25");
assertEq(view[42], 5, "42th value should be 5");
assertEq(view[43], 255, "43th value should be 255");
assertEq(view[44], 148, "44th value should be 148");
assertEq(view[45], 25, "45th value should be 25");
assertEq(view[46], 5, "46th value should be 5");
assertEq(view[47], 255, "47th value should be 255");
assertEq(view[48], 148, "48th value should be 148");
assertEq(view[49], 25, "49th value should be 25");
assertEq(view[50], 5, "50th value should be 5");
assertEq(view[51], 255, "51th value should be 255");
assertEq(view[52], 148, "52th value should be 148");
assertEq(view[53], 25, "53th value should be 25");
assertEq(view[54], 5, "54th value should be 5");
assertEq(view[55], 255, "55th value should be 255");
assertEq(view[56], 148, "56th value should be 148");
assertEq(view[57], 25, "57th value should be 25");
assertEq(view[58], 5, "58th value should be 5");
assertEq(view[59], 255, "59th value should be 255");
assertEq(view[60], 148, "60th value should be 148");
assertEq(view[61], 25, "61th value should be 25");
assertEq(view[62], 5, "62th value should be 5");
assertEq(view[63], 255, "63th value should be 255");
assertEq(view[64], 148, "64th value should be 148");
assertEq(view[65], 25, "65th value should be 25");
assertEq(view[66], 5, "66th value should be 5");
assertEq(view[67], 255, "67th value should be 255");
assertEq(view[68], 148, "68th value should be 148");
assertEq(view[69], 25, "69th value should be 25");
assertEq(view[70], 5, "70th value should be 5");
assertEq(view[71], 255, "71th value should be 255");
assertEq(view[72], 148, "72th value should be 148");
assertEq(view[73], 25, "73th value should be 25");
assertEq(view[74], 5, "74th value should be 5");
assertEq(view[75], 255, "75th value should be 255");
assertEq(view[76], 148, "76th value should be 148");
assertEq(view[77], 25, "77th value should be 25");
assertEq(view[78], 5, "78th value should be 5");
assertEq(view[79], 255, "79th value should be 255");
assertEq(view[80], 148, "80th value should be 148");
assertEq(view[81], 25, "81th value should be 25");
assertEq(view[82], 5, "82th value should be 5");
assertEq(view[83], 255, "83th value should be 255");
assertEq(view[84], 148, "84th value should be 148");
assertEq(view[85], 25, "85th value should be 25");
assertEq(view[86], 5, "86th value should be 5");
assertEq(view[87], 255, "87th value should be 255");
assertEq(view[88], 148, "88th value should be 148");
assertEq(view[89], 25, "89th value should be 25");
assertEq(view[90], 5, "90th value should be 5");
assertEq(view[91], 255, "91th value should be 255");
assertEq(view[92], 148, "92th value should be 148");
assertEq(view[93], 25, "93th value should be 25");
assertEq(view[94], 5, "94th value should be 5");
assertEq(view[95], 255, "95th value should be 255");
assertEq(view[96], 148, "96th value should be 148");
assertEq(view[97], 25, "97th value should be 25");
assertEq(view[98], 5, "98th value should be 5");
assertEq(view[99], 255, "99th value should be 255");
assertEq(view[100], 148, "100th value should be 148");
assertEq(view[101], 25, "101th value should be 25");
assertEq(view[102], 5, "102th value should be 5");
assertEq(view[103], 255, "103th value should be 255");
assertEq(view[104], 148, "104th value should be 148");
assertEq(view[105], 25, "105th value should be 25");
assertEq(view[106], 5, "106th value should be 5");
assertEq(view[107], 255, "107th value should be 255");
assertEq(view[108], 148, "108th value should be 148");
assertEq(view[109], 25, "109th value should be 25");
assertEq(view[110], 5, "110th value should be 5");
assertEq(view[111], 255, "111th value should be 255");
assertEq(view[112], 148, "112th value should be 148");
assertEq(view[113], 25, "113th value should be 25");
assertEq(view[114], 5, "114th value should be 5");
assertEq(view[115], 255, "115th value should be 255");
assertEq(view[116], 148, "116th value should be 148");
assertEq(view[117], 25, "117th value should be 25");
assertEq(view[118], 5, "118th value should be 5");
assertEq(view[119], 255, "119th value should be 255");
assertEq(view[120], 148, "120th value should be 148");
assertEq(view[121], 25, "121th value should be 25");
assertEq(view[122], 5, "122th value should be 5");
assertEq(view[123], 255, "123th value should be 255");
assertEq(view[124], 148, "124th value should be 148");
assertEq(view[125], 25, "125th value should be 25");
assertEq(view[126], 5, "126th value should be 5");
assertEq(view[127], 255, "127th value should be 255");
assertEq(view[128], 148, "128th value should be 148");
assertEq(view[129], 25, "129th value should be 25");
assertEq(view[130], 5, "130th value should be 5");
assertEq(view[131], 255, "131th value should be 255");
assertEq(view[132], 148, "132th value should be 148");
assertEq(view[133], 25, "133th value should be 25");
assertEq(view[134], 5, "134th value should be 5");
assertEq(view[135], 255, "135th value should be 255");
assertEq(view[136], 94, "136th value should be 94");
assertEq(view[137], 166, "137th value should be 166");
assertEq(view[138], 7, "138th value should be 7");
assertEq(view[139], 255, "139th value should be 255");
assertEq(view[140], 94, "140th value should be 94");
assertEq(view[141], 166, "141th value should be 166");
assertEq(view[142], 7, "142th value should be 7");
assertEq(view[143], 255, "143th value should be 255");
assertEq(view[144], 94, "144th value should be 94");
assertEq(view[145], 166, "145th value should be 166");
assertEq(view[146], 7, "146th value should be 7");
assertEq(view[147], 255, "147th value should be 255");
assertEq(view[148], 94, "148th value should be 94");
assertEq(view[149], 166, "149th value should be 166");
assertEq(view[150], 7, "150th value should be 7");
assertEq(view[151], 255, "151th value should be 255");
assertEq(view[152], 94, "152th value should be 94");
assertEq(view[153], 166, "153th value should be 166");
assertEq(view[154], 7, "154th value should be 7");
assertEq(view[155], 255, "155th value should be 255");
assertEq(view[156], 94, "156th value should be 94");
assertEq(view[157], 166, "157th value should be 166");
assertEq(view[158], 7, "158th value should be 7");
assertEq(view[159], 255, "159th value should be 255");
assertEq(view[160], 94, "160th value should be 94");
assertEq(view[161], 166, "161th value should be 166");
assertEq(view[162], 7, "162th value should be 7");
assertEq(view[163], 255, "163th value should be 255");
assertEq(view[164], 94, "164th value should be 94");
assertEq(view[165], 166, "165th value should be 166");
assertEq(view[166], 7, "166th value should be 7");
assertEq(view[167], 255, "167th value should be 255");
assertEq(view[168], 94, "168th value should be 94");
assertEq(view[169], 166, "169th value should be 166");
assertEq(view[170], 7, "170th value should be 7");
assertEq(view[171], 255, "171th value should be 255");
assertEq(view[172], 94, "172th value should be 94");
assertEq(view[173], 166, "173th value should be 166");
assertEq(view[174], 7, "174th value should be 7");
assertEq(view[175], 255, "175th value should be 255");
assertEq(view[176], 94, "176th value should be 94");
assertEq(view[177], 166, "177th value should be 166");
assertEq(view[178], 7, "178th value should be 7");
assertEq(view[179], 255, "179th value should be 255");
assertEq(view[180], 94, "180th value should be 94");
assertEq(view[181], 166, "181th value should be 166");
assertEq(view[182], 7, "182th value should be 7");
assertEq(view[183], 255, "183th value should be 255");
assertEq(view[184], 94, "184th value should be 94");
assertEq(view[185], 166, "185th value should be 166");
assertEq(view[186], 7, "186th value should be 7");
assertEq(view[187], 255, "187th value should be 255");
assertEq(view[188], 41, "188th value should be 41");
assertEq(view[189], 51, "189th value should be 51");
assertEq(view[190], 10, "190th value should be 10");
assertEq(view[191], 255, "191th value should be 255");
assertEq(view[192], 41, "192th value should be 41");
assertEq(view[193], 51, "193th value should be 51");
assertEq(view[194], 10, "194th value should be 10");
assertEq(view[195], 255, "195th value should be 255");
assertEq(view[196], 41, "196th value should be 41");
assertEq(view[197], 51, "197th value should be 51");
assertEq(view[198], 10, "198th value should be 10");
assertEq(view[199], 255, "199th value should be 255");
assertEq(view[200], 41, "200th value should be 41");
assertEq(view[201], 51, "201th value should be 51");
assertEq(view[202], 10, "202th value should be 10");
assertEq(view[203], 255, "203th value should be 255");
assertEq(view[204], 41, "204th value should be 41");
assertEq(view[205], 51, "205th value should be 51");
assertEq(view[206], 10, "206th value should be 10");
assertEq(view[207], 255, "207th value should be 255");
assertEq(view[208], 243, "208th value should be 243");
assertEq(view[209], 191, "209th value should be 191");
assertEq(view[210], 12, "210th value should be 12");
assertEq(view[211], 255, "211th value should be 255");
assertEq(view[212], 243, "212th value should be 243");
assertEq(view[213], 191, "213th value should be 191");
assertEq(view[214], 12, "214th value should be 12");
assertEq(view[215], 255, "215th value should be 255");
assertEq(view[216], 243, "216th value should be 243");
assertEq(view[217], 191, "217th value should be 191");
assertEq(view[218], 12, "218th value should be 12");
assertEq(view[219], 255, "219th value should be 255");
assertEq(view[220], 135, "220th value should be 135");
assertEq(view[221], 217, "221th value should be 217");
assertEq(view[222], 17, "222th value should be 17");
assertEq(view[223], 255, "223th value should be 255");
assertEq(view[224], 176, "224th value should be 176");
assertEq(view[225], 12, "225th value should be 12");
assertEq(view[226], 28, "226th value should be 28");
assertEq(view[227], 255, "227th value should be 255");
assertEq(view[228], 69, "228th value should be 69");
assertEq(view[229], 38, "229th value should be 38");
assertEq(view[230], 33, "230th value should be 33");
assertEq(view[231], 255, "231th value should be 255");
assertEq(view[232], 43, "232th value should be 43");
assertEq(view[233], 166, "233th value should be 166");
assertEq(view[234], 58, "234th value should be 58");
assertEq(view[235], 255, "235th value should be 255");
assertEq(view[236], 179, "236th value should be 179");
assertEq(view[237], 127, "237th value should be 127");
assertEq(view[238], 76, "238th value should be 76");
assertEq(view[239], 255, "239th value should be 255");
assertEq(view[240], 226, "240th value should be 226");
assertEq(view[241], 152, "241th value should be 152");
assertEq(view[242], 183, "242th value should be 183");
assertEq(view[243], 255, "243th value should be 255");
assertEq(view[244], 0, "244th value should be 0");
assertEq(view[245], 0, "245th value should be 0");
assertEq(view[246], 0, "246th value should be 0");
assertEq(view[247], 255, "247th value should be 255");
assertEq(view[248], 0, "248th value should be 0");
assertEq(view[249], 0, "249th value should be 0");
assertEq(view[250], 0, "250th value should be 0");
assertEq(view[251], 255, "251th value should be 255");
assertEq(view[252], 0, "252th value should be 0");
assertEq(view[253], 0, "253th value should be 0");
assertEq(view[254], 0, "254th value should be 0");
assertEq(view[255], 255, "255th value should be 255");
assertEq(view[256], 185, "256th value should be 185");
assertEq(view[257], 101, "257th value should be 101");
assertEq(view[258], 173, "258th value should be 173");
assertEq(view[259], 255, "259th value should be 255");
assertEq(view[260], 2, "260th value should be 2");
assertEq(view[261], 115, "261th value should be 115");
assertEq(view[262], 48, "262th value should be 48");
assertEq(view[263], 255, "263th value should be 255");
assertEq(view[264], 69, "264th value should be 69");
assertEq(view[265], 38, "265th value should be 38");
assertEq(view[266], 33, "266th value should be 33");
assertEq(view[267], 255, "267th value should be 255");
assertEq(view[268], 15, "268th value should be 15");
assertEq(view[269], 179, "269th value should be 179");
assertEq(view[270], 35, "270th value should be 35");
assertEq(view[271], 255, "271th value should be 255");
assertEq(view[272], 82, "272th value should be 82");
assertEq(view[273], 102, "273th value should be 102");
assertEq(view[274], 20, "274th value should be 20");
assertEq(view[275], 255, "275th value should be 255");
assertEq(view[276], 189, "276th value should be 189");
assertEq(view[277], 76, "277th value should be 76");
assertEq(view[278], 15, "278th value should be 15");
assertEq(view[279], 255, "279th value should be 255");
assertEq(view[280], 243, "280th value should be 243");
assertEq(view[281], 191, "281th value should be 191");
assertEq(view[282], 12, "282th value should be 12");
assertEq(view[283], 255, "283th value should be 255");
assertEq(view[284], 41, "284th value should be 41");
assertEq(view[285], 51, "285th value should be 51");
assertEq(view[286], 10, "286th value should be 10");
assertEq(view[287], 255, "287th value should be 255");
assertEq(view[288], 41, "288th value should be 41");
assertEq(view[289], 51, "289th value should be 51");
assertEq(view[290], 10, "290th value should be 10");
assertEq(view[291], 255, "291th value should be 255");
assertEq(view[292], 41, "292th value should be 41");
assertEq(view[293], 51, "293th value should be 51");
assertEq(view[294], 10, "294th value should be 10");
assertEq(view[295], 255, "295th value should be 255");
assertEq(view[296], 41, "296th value should be 41");
assertEq(view[297], 51, "297th value should be 51");
assertEq(view[298], 10, "298th value should be 10");
assertEq(view[299], 255, "299th value should be 255");
assertEq(view[300], 41, "300th value should be 41");
assertEq(view[301], 51, "301th value should be 51");
assertEq(view[302], 10, "302th value should be 10");
assertEq(view[303], 255, "303th value should be 255");
assertEq(view[304], 41, "304th value should be 41");
assertEq(view[305], 51, "305th value should be 51");
assertEq(view[306], 10, "306th value should be 10");
assertEq(view[307], 255, "307th value should be 255");
assertEq(view[308], 94, "308th value should be 94");
assertEq(view[309], 166, "309th value should be 166");
assertEq(view[310], 7, "310th value should be 7");
assertEq(view[311], 255, "311th value should be 255");
assertEq(view[312], 94, "312th value should be 94");
assertEq(view[313], 166, "313th value should be 166");
assertEq(view[314], 7, "314th value should be 7");
assertEq(view[315], 255, "315th value should be 255");
assertEq(view[316], 94, "316th value should be 94");
assertEq(view[317], 166, "317th value should be 166");
assertEq(view[318], 7, "318th value should be 7");
assertEq(view[319], 255, "319th value should be 255");
assertEq(view[320], 94, "320th value should be 94");
assertEq(view[321], 166, "321th value should be 166");
assertEq(view[322], 7, "322th value should be 7");
assertEq(view[323], 255, "323th value should be 255");
assertEq(view[324], 94, "324th value should be 94");
assertEq(view[325], 166, "325th value should be 166");
assertEq(view[326], 7, "326th value should be 7");
assertEq(view[327], 255, "327th value should be 255");
assertEq(view[328], 94, "328th value should be 94");
assertEq(view[329], 166, "329th value should be 166");
assertEq(view[330], 7, "330th value should be 7");
assertEq(view[331], 255, "331th value should be 255");
assertEq(view[332], 148, "332th value should be 148");
assertEq(view[333], 25, "333th value should be 25");
assertEq(view[334], 5, "334th value should be 5");
assertEq(view[335], 255, "335th value should be 255");
assertEq(view[336], 148, "336th value should be 148");
assertEq(view[337], 25, "337th value should be 25");
assertEq(view[338], 5, "338th value should be 5");
assertEq(view[339], 255, "339th value should be 255");
assertEq(view[340], 148, "340th value should be 148");
assertEq(view[341], 25, "341th value should be 25");
assertEq(view[342], 5, "342th value should be 5");
assertEq(view[343], 255, "343th value should be 255");
assertEq(view[344], 148, "344th value should be 148");
assertEq(view[345], 25, "345th value should be 25");
assertEq(view[346], 5, "346th value should be 5");
assertEq(view[347], 255, "347th value should be 255");
assertEq(view[348], 148, "348th value should be 148");
assertEq(view[349], 25, "349th value should be 25");
assertEq(view[350], 5, "350th value should be 5");
assertEq(view[351], 255, "351th value should be 255");
assertEq(view[352], 148, "352th value should be 148");
assertEq(view[353], 25, "353th value should be 25");
assertEq(view[354], 5, "354th value should be 5");
assertEq(view[355], 255, "355th value should be 255");
assertEq(view[356], 148, "356th value should be 148");
assertEq(view[357], 25, "357th value should be 25");
assertEq(view[358], 5, "358th value should be 5");
assertEq(view[359], 255, "359th value should be 255");
assertEq(view[360], 148, "360th value should be 148");
assertEq(view[361], 25, "361th value should be 25");
assertEq(view[362], 5, "362th value should be 5");
assertEq(view[363], 255, "363th value should be 255");
assertEq(view[364], 148, "364th value should be 148");
assertEq(view[365], 25, "365th value should be 25");
assertEq(view[366], 5, "366th value should be 5");
assertEq(view[367], 255, "367th value should be 255");
assertEq(view[368], 202, "368th value should be 202");
assertEq(view[369], 140, "369th value should be 140");
assertEq(view[370], 2, "370th value should be 2");
assertEq(view[371], 255, "371th value should be 255");
assertEq(view[372], 202, "372th value should be 202");
assertEq(view[373], 140, "373th value should be 140");
assertEq(view[374], 2, "374th value should be 2");
assertEq(view[375], 255, "375th value should be 255");
assertEq(view[376], 202, "376th value should be 202");
assertEq(view[377], 140, "377th value should be 140");
assertEq(view[378], 2, "378th value should be 2");
assertEq(view[379], 255, "379th value should be 255");
assertEq(view[380], 202, "380th value should be 202");
assertEq(view[381], 140, "381th value should be 140");
assertEq(view[382], 2, "382th value should be 2");
assertEq(view[383], 255, "383th value should be 255");
assertEq(view[384], 202, "384th value should be 202");
assertEq(view[385], 140, "385th value should be 140");
assertEq(view[386], 2, "386th value should be 2");
assertEq(view[387], 255, "387th value should be 255");
assertEq(view[388], 202, "388th value should be 202");
assertEq(view[389], 140, "389th value should be 140");
assertEq(view[390], 2, "390th value should be 2");
assertEq(view[391], 255, "391th value should be 255");
assertEq(view[392], 202, "392th value should be 202");
assertEq(view[393], 140, "393th value should be 140");
assertEq(view[394], 2, "394th value should be 2");
assertEq(view[395], 255, "395th value should be 255");
assertEq(view[396], 202, "396th value should be 202");
assertEq(view[397], 140, "397th value should be 140");
assertEq(view[398], 2, "398th value should be 2");
assertEq(view[399], 255, "399th value should be 255");
assertEq(view[400], 202, "400th value should be 202");
assertEq(view[401], 140, "401th value should be 140");
assertEq(view[402], 2, "402th value should be 2");
assertEq(view[403], 255, "403th value should be 255");
assertEq(view[404], 202, "404th value should be 202");
assertEq(view[405], 140, "405th value should be 140");
assertEq(view[406], 2, "406th value should be 2");
assertEq(view[407], 255, "407th value should be 255");
assertEq(view[408], 202, "408th value should be 202");
assertEq(view[409], 140, "409th value should be 140");
assertEq(view[410], 2, "410th value should be 2");
assertEq(view[411], 255, "411th value should be 255");
assertEq(view[412], 202, "412th value should be 202");
assertEq(view[413], 140, "413th value should be 140");
assertEq(view[414], 2, "414th value should be 2");
assertEq(view[415], 255, "415th value should be 255");
assertEq(view[416], 202, "416th value should be 202");
assertEq(view[417], 140, "417th value should be 140");
assertEq(view[418], 2, "418th value should be 2");
assertEq(view[419], 255, "419th value should be 255");
assertEq(view[420], 148, "420th value should be 148");
assertEq(view[421], 25, "421th value should be 25");
assertEq(view[422], 5, "422th value should be 5");
assertEq(view[423], 255, "423th value should be 255");
assertEq(view[424], 148, "424th value should be 148");
assertEq(view[425], 25, "425th value should be 25");
assertEq(view[426], 5, "426th value should be 5");
assertEq(view[427], 255, "427th value should be 255");
assertEq(view[428], 148, "428th value should be 148");
assertEq(view[429], 25, "429th value should be 25");
assertEq(view[430], 5, "430th value should be 5");
assertEq(view[431], 255, "431th value should be 255");
assertEq(view[432], 148, "432th value should be 148");
assertEq(view[433], 25, "433th value should be 25");
assertEq(view[434], 5, "434th value should be 5");
assertEq(view[435], 255, "435th value should be 255");
assertEq(view[436], 148, "436th value should be 148");
assertEq(view[437], 25, "437th value should be 25");
assertEq(view[438], 5, "438th value should be 5");
assertEq(view[439], 255, "439th value should be 255");
assertEq(view[440], 148, "440th value should be 148");
assertEq(view[441], 25, "441th value should be 25");
assertEq(view[442], 5, "442th value should be 5");
assertEq(view[443], 255, "443th value should be 255");
assertEq(view[444], 148, "444th value should be 148");
assertEq(view[445], 25, "445th value should be 25");
assertEq(view[446], 5, "446th value should be 5");
assertEq(view[447], 255, "447th value should be 255");
assertEq(view[448], 148, "448th value should be 148");
assertEq(view[449], 25, "449th value should be 25");
assertEq(view[450], 5, "450th value should be 5");
assertEq(view[451], 255, "451th value should be 255");
assertEq(view[452], 148, "452th value should be 148");
assertEq(view[453], 25, "453th value should be 25");
assertEq(view[454], 5, "454th value should be 5");
assertEq(view[455], 255, "455th value should be 255");
assertEq(view[456], 148, "456th value should be 148");
assertEq(view[457], 25, "457th value should be 25");
assertEq(view[458], 5, "458th value should be 5");
assertEq(view[459], 255, "459th value should be 255");
assertEq(view[460], 148, "460th value should be 148");
assertEq(view[461], 25, "461th value should be 25");
assertEq(view[462], 5, "462th value should be 5");
assertEq(view[463], 255, "463th value should be 255");
assertEq(view[464], 148, "464th value should be 148");
assertEq(view[465], 25, "465th value should be 25");
assertEq(view[466], 5, "466th value should be 5");
assertEq(view[467], 255, "467th value should be 255");
assertEq(view[468], 148, "468th value should be 148");
assertEq(view[469], 25, "469th value should be 25");
assertEq(view[470], 5, "470th value should be 5");
assertEq(view[471], 255, "471th value should be 255");
assertEq(view[472], 148, "472th value should be 148");
assertEq(view[473], 25, "473th value should be 25");
assertEq(view[474], 5, "474th value should be 5");
assertEq(view[475], 255, "475th value should be 255");
assertEq(view[476], 148, "476th value should be 148");
assertEq(view[477], 25, "477th value should be 25");
assertEq(view[478], 5, "478th value should be 5");
assertEq(view[479], 255, "479th value should be 255");
assertEq(view[480], 148, "480th value should be 148");
assertEq(view[481], 25, "481th value should be 25");
assertEq(view[482], 5, "482th value should be 5");
assertEq(view[483], 255, "483th value should be 255");
assertEq(view[484], 148, "484th value should be 148");
assertEq(view[485], 25, "485th value should be 25");
assertEq(view[486], 5, "486th value should be 5");
assertEq(view[487], 255, "487th value should be 255");
assertEq(view[488], 148, "488th value should be 148");
assertEq(view[489], 25, "489th value should be 25");
assertEq(view[490], 5, "490th value should be 5");
assertEq(view[491], 255, "491th value should be 255");
assertEq(view[492], 148, "492th value should be 148");
assertEq(view[493], 25, "493th value should be 25");
assertEq(view[494], 5, "494th value should be 5");
assertEq(view[495], 255, "495th value should be 255");
assertEq(view[496], 148, "496th value should be 148");
assertEq(view[497], 25, "497th value should be 25");
assertEq(view[498], 5, "498th value should be 5");
assertEq(view[499], 255, "499th value should be 255");
assertEq(view[500], 148, "500th value should be 148");
assertEq(view[501], 25, "501th value should be 25");
assertEq(view[502], 5, "502th value should be 5");
assertEq(view[503], 255, "503th value should be 255");
assertEq(view[504], 148, "504th value should be 148");
assertEq(view[505], 25, "505th value should be 25");
assertEq(view[506], 5, "506th value should be 5");
assertEq(view[507], 255, "507th value should be 255");
assertEq(view[508], 148, "508th value should be 148");
assertEq(view[509], 25, "509th value should be 25");
assertEq(view[510], 5, "510th value should be 5");
assertEq(view[511], 255, "511th value should be 255");
assertEq(view[512], 148, "512th value should be 148");
assertEq(view[513], 25, "513th value should be 25");
assertEq(view[514], 5, "514th value should be 5");
assertEq(view[515], 255, "515th value should be 255");
assertEq(view[516], 148, "516th value should be 148");
assertEq(view[517], 25, "517th value should be 25");
assertEq(view[518], 5, "518th value should be 5");
assertEq(view[519], 255, "519th value should be 255");
assertEq(view[520], 148, "520th value should be 148");
assertEq(view[521], 25, "521th value should be 25");
assertEq(view[522], 5, "522th value should be 5");
assertEq(view[523], 255, "523th value should be 255");
assertEq(view[524], 148, "524th value should be 148");
assertEq(view[525], 25, "525th value should be 25");
assertEq(view[526], 5, "526th value should be 5");
assertEq(view[527], 255, "527th value should be 255");
assertEq(view[528], 148, "528th value should be 148");
assertEq(view[529], 25, "529th value should be 25");
assertEq(view[530], 5, "530th value should be 5");
assertEq(view[531], 255, "531th value should be 255");
assertEq(view[532], 94, "532th value should be 94");
assertEq(view[533], 166, "533th value should be 166");
assertEq(view[534], 7, "534th value should be 7");
assertEq(view[535], 255, "535th value should be 255");
assertEq(view[536], 94, "536th value should be 94");
assertEq(view[537], 166, "537th value should be 166");
assertEq(view[538], 7, "538th value should be 7");
assertEq(view[539], 255, "539th value should be 255");
assertEq(view[540], 94, "540th value should be 94");
assertEq(view[541], 166, "541th value should be 166");
assertEq(view[542], 7, "542th value should be 7");
assertEq(view[543], 255, "543th value should be 255");
assertEq(view[544], 94, "544th value should be 94");
assertEq(view[545], 166, "545th value should be 166");
assertEq(view[546], 7, "546th value should be 7");
assertEq(view[547], 255, "547th value should be 255");
assertEq(view[548], 94, "548th value should be 94");
assertEq(view[549], 166, "549th value should be 166");
assertEq(view[550], 7, "550th value should be 7");
assertEq(view[551], 255, "551th value should be 255");
assertEq(view[552], 94, "552th value should be 94");
assertEq(view[553], 166, "553th value should be 166");
assertEq(view[554], 7, "554th value should be 7");
assertEq(view[555], 255, "555th value should be 255");
assertEq(view[556], 94, "556th value should be 94");
assertEq(view[557], 166, "557th value should be 166");
assertEq(view[558], 7, "558th value should be 7");
assertEq(view[559], 255, "559th value should be 255");
assertEq(view[560], 94, "560th value should be 94");
assertEq(view[561], 166, "561th value should be 166");
assertEq(view[562], 7, "562th value should be 7");
assertEq(view[563], 255, "563th value should be 255");
assertEq(view[564], 94, "564th value should be 94");
assertEq(view[565], 166, "565th value should be 166");
assertEq(view[566], 7, "566th value should be 7");
assertEq(view[567], 255, "567th value should be 255");
assertEq(view[568], 94, "568th value should be 94");
assertEq(view[569], 166, "569th value should be 166");
assertEq(view[570], 7, "570th value should be 7");
assertEq(view[571], 255, "571th value should be 255");
assertEq(view[572], 94, "572th value should be 94");
assertEq(view[573], 166, "573th value should be 166");
assertEq(view[574], 7, "574th value should be 7");
assertEq(view[575], 255, "575th value should be 255");
assertEq(view[576], 94, "576th value should be 94");
assertEq(view[577], 166, "577th value should be 166");
assertEq(view[578], 7, "578th value should be 7");
assertEq(view[579], 255, "579th value should be 255");
assertEq(view[580], 94, "580th value should be 94");
assertEq(view[581], 166, "581th value should be 166");
assertEq(view[582], 7, "582th value should be 7");
assertEq(view[583], 255, "583th value should be 255");
assertEq(view[584], 41, "584th value should be 41");
assertEq(view[585], 51, "585th value should be 51");
assertEq(view[586], 10, "586th value should be 10");
assertEq(view[587], 255, "587th value should be 255");
assertEq(view[588], 41, "588th value should be 41");
assertEq(view[589], 51, "589th value should be 51");
assertEq(view[590], 10, "590th value should be 10");
assertEq(view[591], 255, "591th value should be 255");
assertEq(view[592], 41, "592th value should be 41");
assertEq(view[593], 51, "593th value should be 51");
assertEq(view[594], 10, "594th value should be 10");
assertEq(view[595], 255, "595th value should be 255");
assertEq(view[596], 41, "596th value should be 41");
assertEq(view[597], 51, "597th value should be 51");
assertEq(view[598], 10, "598th value should be 10");
assertEq(view[599], 255, "599th value should be 255");
assertEq(view[600], 41, "600th value should be 41");
assertEq(view[601], 51, "601th value should be 51");
assertEq(view[602], 10, "602th value should be 10");
assertEq(view[603], 255, "603th value should be 255");
assertEq(view[604], 243, "604th value should be 243");
assertEq(view[605], 191, "605th value should be 191");
assertEq(view[606], 12, "606th value should be 12");
assertEq(view[607], 255, "607th value should be 255");
assertEq(view[608], 243, "608th value should be 243");
assertEq(view[609], 191, "609th value should be 191");
assertEq(view[610], 12, "610th value should be 12");
assertEq(view[611], 255, "611th value should be 255");
assertEq(view[612], 243, "612th value should be 243");
assertEq(view[613], 191, "613th value should be 191");
assertEq(view[614], 12, "614th value should be 12");
assertEq(view[615], 255, "615th value should be 255");
assertEq(view[616], 189, "616th value should be 189");
assertEq(view[617], 76, "617th value should be 76");
assertEq(view[618], 15, "618th value should be 15");
assertEq(view[619], 255, "619th value should be 255");
assertEq(view[620], 135, "620th value should be 135");
assertEq(view[621], 217, "621th value should be 217");
assertEq(view[622], 17, "622th value should be 17");
assertEq(view[623], 255, "623th value should be 255");
assertEq(view[624], 82, "624th value should be 82");
assertEq(view[625], 102, "625th value should be 102");
assertEq(view[626], 20, "626th value should be 20");
assertEq(view[627], 255, "627th value should be 255");
assertEq(view[628], 230, "628th value should be 230");
assertEq(view[629], 127, "629th value should be 127");
assertEq(view[630], 25, "630th value should be 25");
assertEq(view[631], 255, "631th value should be 255");
assertEq(view[632], 167, "632th value should be 167");
assertEq(view[633], 63, "633th value should be 63");
assertEq(view[634], 89, "634th value should be 89");
assertEq(view[635], 255, "635th value should be 255");
assertEq(view[636], 0, "636th value should be 0");
assertEq(view[637], 0, "637th value should be 0");
assertEq(view[638], 0, "638th value should be 0");
assertEq(view[639], 255, "639th value should be 255");
assertEq(view[640], 0, "640th value should be 0");
assertEq(view[641], 0, "641th value should be 0");
assertEq(view[642], 0, "642th value should be 0");
assertEq(view[643], 255, "643th value should be 255");
assertEq(view[644], 0, "644th value should be 0");
assertEq(view[645], 0, "645th value should be 0");
assertEq(view[646], 0, "646th value should be 0");
assertEq(view[647], 255, "647th value should be 255");
assertEq(view[648], 0, "648th value should be 0");
assertEq(view[649], 0, "649th value should be 0");
assertEq(view[650], 0, "650th value should be 0");
assertEq(view[651], 255, "651th value should be 255");
assertEq(view[652], 0, "652th value should be 0");
assertEq(view[653], 0, "653th value should be 0");
assertEq(view[654], 0, "654th value should be 0");
assertEq(view[655], 255, "655th value should be 255");
assertEq(view[656], 0, "656th value should be 0");
assertEq(view[657], 0, "657th value should be 0");
assertEq(view[658], 0, "658th value should be 0");
assertEq(view[659], 255, "659th value should be 255");
assertEq(view[660], 192, "660th value should be 192");
assertEq(view[661], 191, "661th value should be 191");
assertEq(view[662], 63, "662th value should be 63");
assertEq(view[663], 255, "663th value should be 255");
assertEq(view[664], 110, "664th value should be 110");
assertEq(view[665], 89, "665th value should be 89");
assertEq(view[666], 43, "666th value should be 43");
assertEq(view[667], 255, "667th value should be 255");
assertEq(view[668], 164, "668th value should be 164");
assertEq(view[669], 204, "669th value should be 204");
assertEq(view[670], 40, "670th value should be 40");
assertEq(view[671], 255, "671th value should be 255");
assertEq(view[672], 205, "672th value should be 205");
assertEq(view[673], 255, "673th value should be 255");
assertEq(view[674], 50, "674th value should be 50");
assertEq(view[675], 255, "675th value should be 255");
assertEq(view[676], 189, "676th value should be 189");
assertEq(view[677], 76, "677th value should be 76");
assertEq(view[678], 15, "678th value should be 15");
assertEq(view[679], 255, "679th value should be 255");
assertEq(view[680], 243, "680th value should be 243");
assertEq(view[681], 191, "681th value should be 191");
assertEq(view[682], 12, "682th value should be 12");
assertEq(view[683], 255, "683th value should be 255");
assertEq(view[684], 41, "684th value should be 41");
assertEq(view[685], 51, "685th value should be 51");
assertEq(view[686], 10, "686th value should be 10");
assertEq(view[687], 255, "687th value should be 255");
assertEq(view[688], 41, "688th value should be 41");
assertEq(view[689], 51, "689th value should be 51");
assertEq(view[690], 10, "690th value should be 10");
assertEq(view[691], 255, "691th value should be 255");
assertEq(view[692], 41, "692th value should be 41");
assertEq(view[693], 51, "693th value should be 51");
assertEq(view[694], 10, "694th value should be 10");
assertEq(view[695], 255, "695th value should be 255");
assertEq(view[696], 41, "696th value should be 41");
assertEq(view[697], 51, "697th value should be 51");
assertEq(view[698], 10, "698th value should be 10");
assertEq(view[699], 255, "699th value should be 255");
assertEq(view[700], 41, "700th value should be 41");
assertEq(view[701], 51, "701th value should be 51");
assertEq(view[702], 10, "702th value should be 10");
assertEq(view[703], 255, "703th value should be 255");
assertEq(view[704], 41, "704th value should be 41");
assertEq(view[705], 51, "705th value should be 51");
assertEq(view[706], 10, "706th value should be 10");
assertEq(view[707], 255, "707th value should be 255");
assertEq(view[708], 41, "708th value should be 41");
assertEq(view[709], 51, "709th value should be 51");
assertEq(view[710], 10, "710th value should be 10");
assertEq(view[711], 255, "711th value should be 255");
assertEq(view[712], 94, "712th value should be 94");
assertEq(view[713], 166, "713th value should be 166");
assertEq(view[714], 7, "714th value should be 7");
assertEq(view[715], 255, "715th value should be 255");
assertEq(view[716], 94, "716th value should be 94");
assertEq(view[717], 166, "717th value should be 166");
assertEq(view[718], 7, "718th value should be 7");
assertEq(view[719], 255, "719th value should be 255");
assertEq(view[720], 94, "720th value should be 94");
assertEq(view[721], 166, "721th value should be 166");
assertEq(view[722], 7, "722th value should be 7");
assertEq(view[723], 255, "723th value should be 255");
assertEq(view[724], 94, "724th value should be 94");
assertEq(view[725], 166, "725th value should be 166");
assertEq(view[726], 7, "726th value should be 7");
assertEq(view[727], 255, "727th value should be 255");
assertEq(view[728], 94, "728th value should be 94");
assertEq(view[729], 166, "729th value should be 166");
assertEq(view[730], 7, "730th value should be 7");
assertEq(view[731], 255, "731th value should be 255");
assertEq(view[732], 148, "732th value should be 148");
assertEq(view[733], 25, "733th value should be 25");
assertEq(view[734], 5, "734th value should be 5");
assertEq(view[735], 255, "735th value should be 255");
assertEq(view[736], 148, "736th value should be 148");
assertEq(view[737], 25, "737th value should be 25");
assertEq(view[738], 5, "738th value should be 5");
assertEq(view[739], 255, "739th value should be 255");
assertEq(view[740], 148, "740th value should be 148");
assertEq(view[741], 25, "741th value should be 25");
assertEq(view[742], 5, "742th value should be 5");
assertEq(view[743], 255, "743th value should be 255");
assertEq(view[744], 148, "744th value should be 148");
assertEq(view[745], 25, "745th value should be 25");
assertEq(view[746], 5, "746th value should be 5");
assertEq(view[747], 255, "747th value should be 255");
assertEq(view[748], 148, "748th value should be 148");
assertEq(view[749], 25, "749th value should be 25");
assertEq(view[750], 5, "750th value should be 5");
assertEq(view[751], 255, "751th value should be 255");
assertEq(view[752], 148, "752th value should be 148");
assertEq(view[753], 25, "753th value should be 25");
assertEq(view[754], 5, "754th value should be 5");
assertEq(view[755], 255, "755th value should be 255");
assertEq(view[756], 148, "756th value should be 148");
assertEq(view[757], 25, "757th value should be 25");
assertEq(view[758], 5, "758th value should be 5");
assertEq(view[759], 255, "759th value should be 255");
assertEq(view[760], 148, "760th value should be 148");
assertEq(view[761], 25, "761th value should be 25");
assertEq(view[762], 5, "762th value should be 5");
assertEq(view[763], 255, "763th value should be 255");
assertEq(view[764], 148, "764th value should be 148");
assertEq(view[765], 25, "765th value should be 25");
assertEq(view[766], 5, "766th value should be 5");
assertEq(view[767], 255, "767th value should be 255");
assertEq(view[768], 202, "768th value should be 202");
assertEq(view[769], 140, "769th value should be 140");
assertEq(view[770], 2, "770th value should be 2");
assertEq(view[771], 255, "771th value should be 255");
assertEq(view[772], 202, "772th value should be 202");
assertEq(view[773], 140, "773th value should be 140");
assertEq(view[774], 2, "774th value should be 2");
assertEq(view[775], 255, "775th value should be 255");
assertEq(view[776], 202, "776th value should be 202");
assertEq(view[777], 140, "777th value should be 140");
assertEq(view[778], 2, "778th value should be 2");
assertEq(view[779], 255, "779th value should be 255");
assertEq(view[780], 202, "780th value should be 202");
assertEq(view[781], 140, "781th value should be 140");
assertEq(view[782], 2, "782th value should be 2");
assertEq(view[783], 255, "783th value should be 255");
assertEq(view[784], 202, "784th value should be 202");
assertEq(view[785], 140, "785th value should be 140");
assertEq(view[786], 2, "786th value should be 2");
assertEq(view[787], 255, "787th value should be 255");
assertEq(view[788], 202, "788th value should be 202");
assertEq(view[789], 140, "789th value should be 140");
assertEq(view[790], 2, "790th value should be 2");
assertEq(view[791], 255, "791th value should be 255");
assertEq(view[792], 202, "792th value should be 202");
assertEq(view[793], 140, "793th value should be 140");
assertEq(view[794], 2, "794th value should be 2");
assertEq(view[795], 255, "795th value should be 255");
assertEq(view[796], 202, "796th value should be 202");
assertEq(view[797], 140, "797th value should be 140");
assertEq(view[798], 2, "798th value should be 2");
assertEq(view[799], 255, "799th value should be 255");
assertEq(view[800], 202, "800th value should be 202");
assertEq(view[801], 140, "801th value should be 140");
assertEq(view[802], 2, "802th value should be 2");
assertEq(view[803], 255, "803th value should be 255");
assertEq(view[804], 202, "804th value should be 202");
assertEq(view[805], 140, "805th value should be 140");
assertEq(view[806], 2, "806th value should be 2");
assertEq(view[807], 255, "807th value should be 255");
assertEq(view[808], 202, "808th value should be 202");
assertEq(view[809], 140, "809th value should be 140");
assertEq(view[810], 2, "810th value should be 2");
assertEq(view[811], 255, "811th value should be 255");
assertEq(view[812], 202, "812th value should be 202");
assertEq(view[813], 140, "813th value should be 140");
assertEq(view[814], 2, "814th value should be 2");
assertEq(view[815], 255, "815th value should be 255");
assertEq(view[816], 148, "816th value should be 148");
assertEq(view[817], 25, "817th value should be 25");
assertEq(view[818], 5, "818th value should be 5");
assertEq(view[819], 255, "819th value should be 255");
assertEq(view[820], 148, "820th value should be 148");
assertEq(view[821], 25, "821th value should be 25");
assertEq(view[822], 5, "822th value should be 5");
assertEq(view[823], 255, "823th value should be 255");
assertEq(view[824], 148, "824th value should be 148");
assertEq(view[825], 25, "825th value should be 25");
assertEq(view[826], 5, "826th value should be 5");
assertEq(view[827], 255, "827th value should be 255");
assertEq(view[828], 148, "828th value should be 148");
assertEq(view[829], 25, "829th value should be 25");
assertEq(view[830], 5, "830th value should be 5");
assertEq(view[831], 255, "831th value should be 255");
assertEq(view[832], 148, "832th value should be 148");
assertEq(view[833], 25, "833th value should be 25");
assertEq(view[834], 5, "834th value should be 5");
assertEq(view[835], 255, "835th value should be 255");
assertEq(view[836], 148, "836th value should be 148");
assertEq(view[837], 25, "837th value should be 25");
assertEq(view[838], 5, "838th value should be 5");
assertEq(view[839], 255, "839th value should be 255");
assertEq(view[840], 148, "840th value should be 148");
assertEq(view[841], 25, "841th value should be 25");
assertEq(view[842], 5, "842th value should be 5");
assertEq(view[843], 255, "843th value should be 255");
assertEq(view[844], 148, "844th value should be 148");
assertEq(view[845], 25, "845th value should be 25");
assertEq(view[846], 5, "846th value should be 5");
assertEq(view[847], 255, "847th value should be 255");
assertEq(view[848], 148, "848th value should be 148");
assertEq(view[849], 25, "849th value should be 25");
assertEq(view[850], 5, "850th value should be 5");
assertEq(view[851], 255, "851th value should be 255");
assertEq(view[852], 148, "852th value should be 148");
assertEq(view[853], 25, "853th value should be 25");
assertEq(view[854], 5, "854th value should be 5");
assertEq(view[855], 255, "855th value should be 255");
assertEq(view[856], 148, "856th value should be 148");
assertEq(view[857], 25, "857th value should be 25");
assertEq(view[858], 5, "858th value should be 5");
assertEq(view[859], 255, "859th value should be 255");
assertEq(view[860], 148, "860th value should be 148");
assertEq(view[861], 25, "861th value should be 25");
assertEq(view[862], 5, "862th value should be 5");
assertEq(view[863], 255, "863th value should be 255");
assertEq(view[864], 148, "864th value should be 148");
assertEq(view[865], 25, "865th value should be 25");
assertEq(view[866], 5, "866th value should be 5");
assertEq(view[867], 255, "867th value should be 255");
assertEq(view[868], 148, "868th value should be 148");
assertEq(view[869], 25, "869th value should be 25");
assertEq(view[870], 5, "870th value should be 5");
assertEq(view[871], 255, "871th value should be 255");
assertEq(view[872], 148, "872th value should be 148");
assertEq(view[873], 25, "873th value should be 25");
assertEq(view[874], 5, "874th value should be 5");
assertEq(view[875], 255, "875th value should be 255");
assertEq(view[876], 148, "876th value should be 148");
assertEq(view[877], 25, "877th value should be 25");
assertEq(view[878], 5, "878th value should be 5");
assertEq(view[879], 255, "879th value should be 255");
assertEq(view[880], 148, "880th value should be 148");
assertEq(view[881], 25, "881th value should be 25");
assertEq(view[882], 5, "882th value should be 5");
assertEq(view[883], 255, "883th value should be 255");
assertEq(view[884], 148, "884th value should be 148");
assertEq(view[885], 25, "885th value should be 25");
assertEq(view[886], 5, "886th value should be 5");
assertEq(view[887], 255, "887th value should be 255");
assertEq(view[888], 148, "888th value should be 148");
assertEq(view[889], 25, "889th value should be 25");
assertEq(view[890], 5, "890th value should be 5");
assertEq(view[891], 255, "891th value should be 255");
assertEq(view[892], 148, "892th value should be 148");
assertEq(view[893], 25, "893th value should be 25");
assertEq(view[894], 5, "894th value should be 5");
assertEq(view[895], 255, "895th value should be 255");
assertEq(view[896], 148, "896th value should be 148");
assertEq(view[897], 25, "897th value should be 25");
assertEq(view[898], 5, "898th value should be 5");
assertEq(view[899], 255, "899th value should be 255");
assertEq(view[900], 148, "900th value should be 148");
assertEq(view[901], 25, "901th value should be 25");
assertEq(view[902], 5, "902th value should be 5");
assertEq(view[903], 255, "903th value should be 255");
assertEq(view[904], 148, "904th value should be 148");
assertEq(view[905], 25, "905th value should be 25");
assertEq(view[906], 5, "906th value should be 5");
assertEq(view[907], 255, "907th value should be 255");
assertEq(view[908], 148, "908th value should be 148");
assertEq(view[909], 25, "909th value should be 25");
assertEq(view[910], 5, "910th value should be 5");
assertEq(view[911], 255, "911th value should be 255");
assertEq(view[912], 148, "912th value should be 148");
assertEq(view[913], 25, "913th value should be 25");
assertEq(view[914], 5, "914th value should be 5");
assertEq(view[915], 255, "915th value should be 255");
assertEq(view[916], 148, "916th value should be 148");
assertEq(view[917], 25, "917th value should be 25");
assertEq(view[918], 5, "918th value should be 5");
assertEq(view[919], 255, "919th value should be 255");
assertEq(view[920], 148, "920th value should be 148");
assertEq(view[921], 25, "921th value should be 25");
assertEq(view[922], 5, "922th value should be 5");
assertEq(view[923], 255, "923th value should be 255");
assertEq(view[924], 148, "924th value should be 148");
assertEq(view[925], 25, "925th value should be 25");
assertEq(view[926], 5, "926th value should be 5");
assertEq(view[927], 255, "927th value should be 255");
assertEq(view[928], 148, "928th value should be 148");
assertEq(view[929], 25, "929th value should be 25");
assertEq(view[930], 5, "930th value should be 5");
assertEq(view[931], 255, "931th value should be 255");
assertEq(view[932], 94, "932th value should be 94");
assertEq(view[933], 166, "933th value should be 166");
assertEq(view[934], 7, "934th value should be 7");
assertEq(view[935], 255, "935th value should be 255");
assertEq(view[936], 94, "936th value should be 94");
assertEq(view[937], 166, "937th value should be 166");
assertEq(view[938], 7, "938th value should be 7");
assertEq(view[939], 255, "939th value should be 255");
assertEq(view[940], 94, "940th value should be 94");
assertEq(view[941], 166, "941th value should be 166");
assertEq(view[942], 7, "942th value should be 7");
assertEq(view[943], 255, "943th value should be 255");
assertEq(view[944], 94, "944th value should be 94");
assertEq(view[945], 166, "945th value should be 166");
assertEq(view[946], 7, "946th value should be 7");
assertEq(view[947], 255, "947th value should be 255");
assertEq(view[948], 94, "948th value should be 94");
assertEq(view[949], 166, "949th value should be 166");
assertEq(view[950], 7, "950th value should be 7");
assertEq(view[951], 255, "951th value should be 255");
assertEq(view[952], 94, "952th value should be 94");
assertEq(view[953], 166, "953th value should be 166");
assertEq(view[954], 7, "954th value should be 7");
assertEq(view[955], 255, "955th value should be 255");
assertEq(view[956], 94, "956th value should be 94");
assertEq(view[957], 166, "957th value should be 166");
assertEq(view[958], 7, "958th value should be 7");
assertEq(view[959], 255, "959th value should be 255");
assertEq(view[960], 94, "960th value should be 94");
assertEq(view[961], 166, "961th value should be 166");
assertEq(view[962], 7, "962th value should be 7");
assertEq(view[963], 255, "963th value should be 255");
assertEq(view[964], 94, "964th value should be 94");
assertEq(view[965], 166, "965th value should be 166");
assertEq(view[966], 7, "966th value should be 7");
assertEq(view[967], 255, "967th value should be 255");
assertEq(view[968], 94, "968th value should be 94");
assertEq(view[969], 166, "969th value should be 166");
assertEq(view[970], 7, "970th value should be 7");
assertEq(view[971], 255, "971th value should be 255");
assertEq(view[972], 94, "972th value should be 94");
assertEq(view[973], 166, "973th value should be 166");
assertEq(view[974], 7, "974th value should be 7");
assertEq(view[975], 255, "975th value should be 255");
assertEq(view[976], 94, "976th value should be 94");
assertEq(view[977], 166, "977th value should be 166");
assertEq(view[978], 7, "978th value should be 7");
assertEq(view[979], 255, "979th value should be 255");
assertEq(view[980], 94, "980th value should be 94");
assertEq(view[981], 166, "981th value should be 166");
assertEq(view[982], 7, "982th value should be 7");
assertEq(view[983], 255, "983th value should be 255");
assertEq(view[984], 41, "984th value should be 41");
assertEq(view[985], 51, "985th value should be 51");
assertEq(view[986], 10, "986th value should be 10");
assertEq(view[987], 255, "987th value should be 255");
assertEq(view[988], 41, "988th value should be 41");
assertEq(view[989], 51, "989th value should be 51");
assertEq(view[990], 10, "990th value should be 10");
assertEq(view[991], 255, "991th value should be 255");
assertEq(view[992], 41, "992th value should be 41");
assertEq(view[993], 51, "993th value should be 51");
assertEq(view[994], 10, "994th value should be 10");
assertEq(view[995], 255, "995th value should be 255");
assertEq(view[996], 243, "996th value should be 243");
assertEq(view[997], 191, "997th value should be 191");
assertEq(view[998], 12, "998th value should be 12");
assertEq(view[999], 255, "999th value should be 255");
assertEq(view[1000], 243, "1000th value should be 243");
assertEq(view[1001], 191, "1001th value should be 191");
assertEq(view[1002], 12, "1002th value should be 12");
assertEq(view[1003], 255, "1003th value should be 255");
assertEq(view[1004], 243, "1004th value should be 243");
assertEq(view[1005], 191, "1005th value should be 191");
assertEq(view[1006], 12, "1006th value should be 12");
assertEq(view[1007], 255, "1007th value should be 255");
assertEq(view[1008], 243, "1008th value should be 243");
assertEq(view[1009], 191, "1009th value should be 191");
assertEq(view[1010], 12, "1010th value should be 12");
assertEq(view[1011], 255, "1011th value should be 255");
assertEq(view[1012], 189, "1012th value should be 189");
assertEq(view[1013], 76, "1013th value should be 76");
assertEq(view[1014], 15, "1014th value should be 15");
assertEq(view[1015], 255, "1015th value should be 255");
assertEq(view[1016], 189, "1016th value should be 189");
assertEq(view[1017], 76, "1017th value should be 76");
assertEq(view[1018], 15, "1018th value should be 15");
assertEq(view[1019], 255, "1019th value should be 255");
assertEq(view[1020], 135, "1020th value should be 135");
assertEq(view[1021], 217, "1021th value should be 217");
assertEq(view[1022], 17, "1022th value should be 17");
assertEq(view[1023], 255, "1023th value should be 255");
assertEq(view[1024], 82, "1024th value should be 82");
assertEq(view[1025], 102, "1025th value should be 102");
assertEq(view[1026], 20, "1026th value should be 20");
assertEq(view[1027], 255, "1027th value should be 255");
assertEq(view[1028], 230, "1028th value should be 230");
assertEq(view[1029], 127, "1029th value should be 127");
assertEq(view[1030], 25, "1030th value should be 25");
assertEq(view[1031], 255, "1031th value should be 255");
assertEq(view[1032], 5, "1032th value should be 5");
assertEq(view[1033], 230, "1033th value should be 230");
assertEq(view[1034], 96, "1034th value should be 96");
assertEq(view[1035], 255, "1035th value should be 255");
assertEq(view[1036], 0, "1036th value should be 0");
assertEq(view[1037], 0, "1037th value should be 0");
assertEq(view[1038], 0, "1038th value should be 0");
assertEq(view[1039], 255, "1039th value should be 255");
assertEq(view[1040], 0, "1040th value should be 0");
assertEq(view[1041], 0, "1041th value should be 0");
assertEq(view[1042], 0, "1042th value should be 0");
assertEq(view[1043], 255, "1043th value should be 255");
assertEq(view[1044], 0, "1044th value should be 0");
assertEq(view[1045], 0, "1045th value should be 0");
assertEq(view[1046], 0, "1046th value should be 0");
assertEq(view[1047], 255, "1047th value should be 255");
assertEq(view[1048], 0, "1048th value should be 0");
assertEq(view[1049], 0, "1049th value should be 0");
assertEq(view[1050], 0, "1050th value should be 0");
assertEq(view[1051], 255, "1051th value should be 255");
assertEq(view[1052], 0, "1052th value should be 0");
assertEq(view[1053], 0, "1053th value should be 0");
assertEq(view[1054], 0, "1054th value should be 0");
assertEq(view[1055], 255, "1055th value should be 255");
assertEq(view[1056], 0, "1056th value should be 0");
assertEq(view[1057], 0, "1057th value should be 0");
assertEq(view[1058], 0, "1058th value should be 0");
assertEq(view[1059], 255, "1059th value should be 255");
assertEq(view[1060], 0, "1060th value should be 0");
assertEq(view[1061], 0, "1061th value should be 0");
assertEq(view[1062], 0, "1062th value should be 0");
assertEq(view[1063], 255, "1063th value should be 255");
assertEq(view[1064], 78, "1064th value should be 78");
assertEq(view[1065], 127, "1065th value should be 127");
assertEq(view[1066], 178, "1066th value should be 178");
assertEq(view[1067], 255, "1067th value should be 255");
assertEq(view[1068], 69, "1068th value should be 69");
assertEq(view[1069], 38, "1069th value should be 38");
assertEq(view[1070], 33, "1070th value should be 33");
assertEq(view[1071], 255, "1071th value should be 255");
assertEq(view[1072], 82, "1072th value should be 82");
assertEq(view[1073], 102, "1073th value should be 102");
assertEq(view[1074], 20, "1074th value should be 20");
assertEq(view[1075], 255, "1075th value should be 255");
assertEq(view[1076], 189, "1076th value should be 189");
assertEq(view[1077], 76, "1077th value should be 76");
assertEq(view[1078], 15, "1078th value should be 15");
assertEq(view[1079], 255, "1079th value should be 255");
assertEq(view[1080], 189, "1080th value should be 189");
assertEq(view[1081], 76, "1081th value should be 76");
assertEq(view[1082], 15, "1082th value should be 15");
assertEq(view[1083], 255, "1083th value should be 255");
assertEq(view[1084], 243, "1084th value should be 243");
assertEq(view[1085], 191, "1085th value should be 191");
assertEq(view[1086], 12, "1086th value should be 12");
assertEq(view[1087], 255, "1087th value should be 255");
assertEq(view[1088], 41, "1088th value should be 41");
assertEq(view[1089], 51, "1089th value should be 51");
assertEq(view[1090], 10, "1090th value should be 10");
assertEq(view[1091], 255, "1091th value should be 255");
assertEq(view[1092], 41, "1092th value should be 41");
assertEq(view[1093], 51, "1093th value should be 51");
assertEq(view[1094], 10, "1094th value should be 10");
assertEq(view[1095], 255, "1095th value should be 255");
assertEq(view[1096], 41, "1096th value should be 41");
assertEq(view[1097], 51, "1097th value should be 51");
assertEq(view[1098], 10, "1098th value should be 10");
assertEq(view[1099], 255, "1099th value should be 255");
assertEq(view[1100], 41, "1100th value should be 41");
assertEq(view[1101], 51, "1101th value should be 51");
assertEq(view[1102], 10, "1102th value should be 10");
assertEq(view[1103], 255, "1103th value should be 255");
assertEq(view[1104], 41, "1104th value should be 41");
assertEq(view[1105], 51, "1105th value should be 51");
assertEq(view[1106], 10, "1106th value should be 10");
assertEq(view[1107], 255, "1107th value should be 255");
assertEq(view[1108], 41, "1108th value should be 41");
assertEq(view[1109], 51, "1109th value should be 51");
assertEq(view[1110], 10, "1110th value should be 10");
assertEq(view[1111], 255, "1111th value should be 255");
assertEq(view[1112], 41, "1112th value should be 41");
assertEq(view[1113], 51, "1113th value should be 51");
assertEq(view[1114], 10, "1114th value should be 10");
assertEq(view[1115], 255, "1115th value should be 255");
assertEq(view[1116], 94, "1116th value should be 94");
assertEq(view[1117], 166, "1117th value should be 166");
assertEq(view[1118], 7, "1118th value should be 7");
assertEq(view[1119], 255, "1119th value should be 255");
assertEq(view[1120], 94, "1120th value should be 94");
assertEq(view[1121], 166, "1121th value should be 166");
assertEq(view[1122], 7, "1122th value should be 7");
assertEq(view[1123], 255, "1123th value should be 255");
assertEq(view[1124], 94, "1124th value should be 94");
assertEq(view[1125], 166, "1125th value should be 166");
assertEq(view[1126], 7, "1126th value should be 7");
assertEq(view[1127], 255, "1127th value should be 255");
assertEq(view[1128], 94, "1128th value should be 94");
assertEq(view[1129], 166, "1129th value should be 166");
assertEq(view[1130], 7, "1130th value should be 7");
assertEq(view[1131], 255, "1131th value should be 255");
assertEq(view[1132], 94, "1132th value should be 94");
assertEq(view[1133], 166, "1133th value should be 166");
assertEq(view[1134], 7, "1134th value should be 7");
assertEq(view[1135], 255, "1135th value should be 255");
assertEq(view[1136], 148, "1136th value should be 148");
assertEq(view[1137], 25, "1137th value should be 25");
assertEq(view[1138], 5, "1138th value should be 5");
assertEq(view[1139], 255, "1139th value should be 255");
assertEq(view[1140], 148, "1140th value should be 148");
assertEq(view[1141], 25, "1141th value should be 25");
assertEq(view[1142], 5, "1142th value should be 5");
assertEq(view[1143], 255, "1143th value should be 255");
assertEq(view[1144], 148, "1144th value should be 148");
assertEq(view[1145], 25, "1145th value should be 25");
assertEq(view[1146], 5, "1146th value should be 5");
assertEq(view[1147], 255, "1147th value should be 255");
assertEq(view[1148], 148, "1148th value should be 148");
assertEq(view[1149], 25, "1149th value should be 25");
assertEq(view[1150], 5, "1150th value should be 5");
assertEq(view[1151], 255, "1151th value should be 255");
assertEq(view[1152], 148, "1152th value should be 148");
assertEq(view[1153], 25, "1153th value should be 25");
assertEq(view[1154], 5, "1154th value should be 5");
assertEq(view[1155], 255, "1155th value should be 255");
assertEq(view[1156], 148, "1156th value should be 148");
assertEq(view[1157], 25, "1157th value should be 25");
assertEq(view[1158], 5, "1158th value should be 5");
assertEq(view[1159], 255, "1159th value should be 255");
assertEq(view[1160], 148, "1160th value should be 148");
assertEq(view[1161], 25, "1161th value should be 25");
assertEq(view[1162], 5, "1162th value should be 5");
assertEq(view[1163], 255, "1163th value should be 255");
assertEq(view[1164], 148, "1164th value should be 148");
assertEq(view[1165], 25, "1165th value should be 25");
assertEq(view[1166], 5, "1166th value should be 5");
assertEq(view[1167], 255, "1167th value should be 255");
assertEq(view[1168], 148, "1168th value should be 148");
assertEq(view[1169], 25, "1169th value should be 25");
assertEq(view[1170], 5, "1170th value should be 5");
assertEq(view[1171], 255, "1171th value should be 255");
assertEq(view[1172], 202, "1172th value should be 202");
assertEq(view[1173], 140, "1173th value should be 140");
assertEq(view[1174], 2, "1174th value should be 2");
assertEq(view[1175], 255, "1175th value should be 255");
assertEq(view[1176], 202, "1176th value should be 202");
assertEq(view[1177], 140, "1177th value should be 140");
assertEq(view[1178], 2, "1178th value should be 2");
assertEq(view[1179], 255, "1179th value should be 255");
assertEq(view[1180], 202, "1180th value should be 202");
assertEq(view[1181], 140, "1181th value should be 140");
assertEq(view[1182], 2, "1182th value should be 2");
assertEq(view[1183], 255, "1183th value should be 255");
assertEq(view[1184], 202, "1184th value should be 202");
assertEq(view[1185], 140, "1185th value should be 140");
assertEq(view[1186], 2, "1186th value should be 2");
assertEq(view[1187], 255, "1187th value should be 255");
assertEq(view[1188], 202, "1188th value should be 202");
assertEq(view[1189], 140, "1189th value should be 140");
assertEq(view[1190], 2, "1190th value should be 2");
assertEq(view[1191], 255, "1191th value should be 255");
assertEq(view[1192], 202, "1192th value should be 202");
assertEq(view[1193], 140, "1193th value should be 140");
assertEq(view[1194], 2, "1194th value should be 2");
assertEq(view[1195], 255, "1195th value should be 255");
assertEq(view[1196], 202, "1196th value should be 202");
assertEq(view[1197], 140, "1197th value should be 140");
assertEq(view[1198], 2, "1198th value should be 2");
assertEq(view[1199], 255, "1199th value should be 255");
assertEq(view[1200], 202, "1200th value should be 202");
assertEq(view[1201], 140, "1201th value should be 140");
assertEq(view[1202], 2, "1202th value should be 2");
assertEq(view[1203], 255, "1203th value should be 255");
assertEq(view[1204], 202, "1204th value should be 202");
assertEq(view[1205], 140, "1205th value should be 140");
assertEq(view[1206], 2, "1206th value should be 2");
assertEq(view[1207], 255, "1207th value should be 255");
assertEq(view[1208], 202, "1208th value should be 202");
assertEq(view[1209], 140, "1209th value should be 140");
assertEq(view[1210], 2, "1210th value should be 2");
assertEq(view[1211], 255, "1211th value should be 255");
assertEq(view[1212], 202, "1212th value should be 202");
assertEq(view[1213], 140, "1213th value should be 140");
assertEq(view[1214], 2, "1214th value should be 2");
assertEq(view[1215], 255, "1215th value should be 255");
assertEq(view[1216], 148, "1216th value should be 148");
assertEq(view[1217], 25, "1217th value should be 25");
assertEq(view[1218], 5, "1218th value should be 5");
assertEq(view[1219], 255, "1219th value should be 255");
assertEq(view[1220], 148, "1220th value should be 148");
assertEq(view[1221], 25, "1221th value should be 25");
assertEq(view[1222], 5, "1222th value should be 5");
assertEq(view[1223], 255, "1223th value should be 255");
assertEq(view[1224], 148, "1224th value should be 148");
assertEq(view[1225], 25, "1225th value should be 25");
assertEq(view[1226], 5, "1226th value should be 5");
assertEq(view[1227], 255, "1227th value should be 255");
assertEq(view[1228], 148, "1228th value should be 148");
assertEq(view[1229], 25, "1229th value should be 25");
assertEq(view[1230], 5, "1230th value should be 5");
assertEq(view[1231], 255, "1231th value should be 255");
assertEq(view[1232], 148, "1232th value should be 148");
assertEq(view[1233], 25, "1233th value should be 25");
assertEq(view[1234], 5, "1234th value should be 5");
assertEq(view[1235], 255, "1235th value should be 255");
assertEq(view[1236], 148, "1236th value should be 148");
assertEq(view[1237], 25, "1237th value should be 25");
assertEq(view[1238], 5, "1238th value should be 5");
assertEq(view[1239], 255, "1239th value should be 255");
assertEq(view[1240], 148, "1240th value should be 148");
assertEq(view[1241], 25, "1241th value should be 25");
assertEq(view[1242], 5, "1242th value should be 5");
assertEq(view[1243], 255, "1243th value should be 255");
assertEq(view[1244], 148, "1244th value should be 148");
assertEq(view[1245], 25, "1245th value should be 25");
assertEq(view[1246], 5, "1246th value should be 5");
assertEq(view[1247], 255, "1247th value should be 255");
assertEq(view[1248], 148, "1248th value should be 148");
assertEq(view[1249], 25, "1249th value should be 25");
assertEq(view[1250], 5, "1250th value should be 5");
assertEq(view[1251], 255, "1251th value should be 255");
assertEq(view[1252], 148, "1252th value should be 148");
assertEq(view[1253], 25, "1253th value should be 25");
assertEq(view[1254], 5, "1254th value should be 5");
assertEq(view[1255], 255, "1255th value should be 255");
assertEq(view[1256], 148, "1256th value should be 148");
assertEq(view[1257], 25, "1257th value should be 25");
assertEq(view[1258], 5, "1258th value should be 5");
assertEq(view[1259], 255, "1259th value should be 255");
assertEq(view[1260], 148, "1260th value should be 148");
assertEq(view[1261], 25, "1261th value should be 25");
assertEq(view[1262], 5, "1262th value should be 5");
assertEq(view[1263], 255, "1263th value should be 255");
assertEq(view[1264], 148, "1264th value should be 148");
assertEq(view[1265], 25, "1265th value should be 25");
assertEq(view[1266], 5, "1266th value should be 5");
assertEq(view[1267], 255, "1267th value should be 255");
assertEq(view[1268], 148, "1268th value should be 148");
assertEq(view[1269], 25, "1269th value should be 25");
assertEq(view[1270], 5, "1270th value should be 5");
assertEq(view[1271], 255, "1271th value should be 255");
assertEq(view[1272], 148, "1272th value should be 148");
assertEq(view[1273], 25, "1273th value should be 25");
assertEq(view[1274], 5, "1274th value should be 5");
assertEq(view[1275], 255, "1275th value should be 255");
assertEq(view[1276], 148, "1276th value should be 148");
assertEq(view[1277], 25, "1277th value should be 25");
assertEq(view[1278], 5, "1278th value should be 5");
assertEq(view[1279], 255, "1279th value should be 255");
assertEq(view[1280], 148, "1280th value should be 148");
assertEq(view[1281], 25, "1281th value should be 25");
assertEq(view[1282], 5, "1282th value should be 5");
assertEq(view[1283], 255, "1283th value should be 255");
assertEq(view[1284], 148, "1284th value should be 148");
assertEq(view[1285], 25, "1285th value should be 25");
assertEq(view[1286], 5, "1286th value should be 5");
assertEq(view[1287], 255, "1287th value should be 255");
assertEq(view[1288], 148, "1288th value should be 148");
assertEq(view[1289], 25, "1289th value should be 25");
assertEq(view[1290], 5, "1290th value should be 5");
assertEq(view[1291], 255, "1291th value should be 255");
assertEq(view[1292], 148, "1292th value should be 148");
assertEq(view[1293], 25, "1293th value should be 25");
assertEq(view[1294], 5, "1294th value should be 5");
assertEq(view[1295], 255, "1295th value should be 255");
assertEq(view[1296], 148, "1296th value should be 148");
assertEq(view[1297], 25, "1297th value should be 25");
assertEq(view[1298], 5, "1298th value should be 5");
assertEq(view[1299], 255, "1299th value should be 255");
assertEq(view[1300], 148, "1300th value should be 148");
assertEq(view[1301], 25, "1301th value should be 25");
assertEq(view[1302], 5, "1302th value should be 5");
assertEq(view[1303], 255, "1303th value should be 255");
assertEq(view[1304], 148, "1304th value should be 148");
assertEq(view[1305], 25, "1305th value should be 25");
assertEq(view[1306], 5, "1306th value should be 5");
assertEq(view[1307], 255, "1307th value should be 255");
assertEq(view[1308], 148, "1308th value should be 148");
assertEq(view[1309], 25, "1309th value should be 25");
assertEq(view[1310], 5, "1310th value should be 5");
assertEq(view[1311], 255, "1311th value should be 255");
assertEq(view[1312], 148, "1312th value should be 148");
assertEq(view[1313], 25, "1313th value should be 25");
assertEq(view[1314], 5, "1314th value should be 5");
assertEq(view[1315], 255, "1315th value should be 255");
assertEq(view[1316], 148, "1316th value should be 148");
assertEq(view[1317], 25, "1317th value should be 25");
assertEq(view[1318], 5, "1318th value should be 5");
assertEq(view[1319], 255, "1319th value should be 255");
assertEq(view[1320], 148, "1320th value should be 148");
assertEq(view[1321], 25, "1321th value should be 25");
assertEq(view[1322], 5, "1322th value should be 5");
assertEq(view[1323], 255, "1323th value should be 255");
assertEq(view[1324], 148, "1324th value should be 148");
assertEq(view[1325], 25, "1325th value should be 25");
assertEq(view[1326], 5, "1326th value should be 5");
assertEq(view[1327], 255, "1327th value should be 255");
assertEq(view[1328], 94, "1328th value should be 94");
assertEq(view[1329], 166, "1329th value should be 166");
assertEq(view[1330], 7, "1330th value should be 7");
assertEq(view[1331], 255, "1331th value should be 255");
assertEq(view[1332], 94, "1332th value should be 94");
assertEq(view[1333], 166, "1333th value should be 166");
assertEq(view[1334], 7, "1334th value should be 7");
assertEq(view[1335], 255, "1335th value should be 255");
assertEq(view[1336], 94, "1336th value should be 94");
assertEq(view[1337], 166, "1337th value should be 166");
assertEq(view[1338], 7, "1338th value should be 7");
assertEq(view[1339], 255, "1339th value should be 255");
assertEq(view[1340], 94, "1340th value should be 94");
assertEq(view[1341], 166, "1341th value should be 166");
assertEq(view[1342], 7, "1342th value should be 7");
assertEq(view[1343], 255, "1343th value should be 255");
assertEq(view[1344], 94, "1344th value should be 94");
assertEq(view[1345], 166, "1345th value should be 166");
assertEq(view[1346], 7, "1346th value should be 7");
assertEq(view[1347], 255, "1347th value should be 255");
assertEq(view[1348], 94, "1348th value should be 94");
assertEq(view[1349], 166, "1349th value should be 166");
assertEq(view[1350], 7, "1350th value should be 7");
assertEq(view[1351], 255, "1351th value should be 255");
assertEq(view[1352], 94, "1352th value should be 94");
assertEq(view[1353], 166, "1353th value should be 166");
assertEq(view[1354], 7, "1354th value should be 7");
assertEq(view[1355], 255, "1355th value should be 255");
assertEq(view[1356], 94, "1356th value should be 94");
assertEq(view[1357], 166, "1357th value should be 166");
assertEq(view[1358], 7, "1358th value should be 7");
assertEq(view[1359], 255, "1359th value should be 255");
assertEq(view[1360], 94, "1360th value should be 94");
assertEq(view[1361], 166, "1361th value should be 166");
assertEq(view[1362], 7, "1362th value should be 7");
assertEq(view[1363], 255, "1363th value should be 255");
assertEq(view[1364], 94, "1364th value should be 94");
assertEq(view[1365], 166, "1365th value should be 166");
assertEq(view[1366], 7, "1366th value should be 7");
assertEq(view[1367], 255, "1367th value should be 255");
assertEq(view[1368], 94, "1368th value should be 94");
assertEq(view[1369], 166, "1369th value should be 166");
assertEq(view[1370], 7, "1370th value should be 7");
assertEq(view[1371], 255, "1371th value should be 255");
assertEq(view[1372], 94, "1372th value should be 94");
assertEq(view[1373], 166, "1373th value should be 166");
assertEq(view[1374], 7, "1374th value should be 7");
assertEq(view[1375], 255, "1375th value should be 255");
assertEq(view[1376], 94, "1376th value should be 94");
assertEq(view[1377], 166, "1377th value should be 166");
assertEq(view[1378], 7, "1378th value should be 7");
assertEq(view[1379], 255, "1379th value should be 255");
assertEq(view[1380], 41, "1380th value should be 41");
assertEq(view[1381], 51, "1381th value should be 51");
assertEq(view[1382], 10, "1382th value should be 10");
assertEq(view[1383], 255, "1383th value should be 255");
assertEq(view[1384], 41, "1384th value should be 41");
assertEq(view[1385], 51, "1385th value should be 51");
assertEq(view[1386], 10, "1386th value should be 10");
assertEq(view[1387], 255, "1387th value should be 255");
assertEq(view[1388], 41, "1388th value should be 41");
assertEq(view[1389], 51, "1389th value should be 51");
assertEq(view[1390], 10, "1390th value should be 10");
assertEq(view[1391], 255, "1391th value should be 255");
assertEq(view[1392], 243, "1392th value should be 243");
assertEq(view[1393], 191, "1393th value should be 191");
assertEq(view[1394], 12, "1394th value should be 12");
assertEq(view[1395], 255, "1395th value should be 255");
assertEq(view[1396], 243, "1396th value should be 243");
assertEq(view[1397], 191, "1397th value should be 191");
assertEq(view[1398], 12, "1398th value should be 12");
assertEq(view[1399], 255, "1399th value should be 255");
assertEq(view[1400], 243, "1400th value should be 243");
assertEq(view[1401], 191, "1401th value should be 191");
assertEq(view[1402], 12, "1402th value should be 12");
assertEq(view[1403], 255, "1403th value should be 255");
assertEq(view[1404], 243, "1404th value should be 243");
assertEq(view[1405], 191, "1405th value should be 191");
assertEq(view[1406], 12, "1406th value should be 12");
assertEq(view[1407], 255, "1407th value should be 255");
assertEq(view[1408], 243, "1408th value should be 243");
assertEq(view[1409], 191, "1409th value should be 191");
assertEq(view[1410], 12, "1410th value should be 12");
assertEq(view[1411], 255, "1411th value should be 255");
assertEq(view[1412], 189, "1412th value should be 189");
assertEq(view[1413], 76, "1413th value should be 76");
assertEq(view[1414], 15, "1414th value should be 15");
assertEq(view[1415], 255, "1415th value should be 255");
assertEq(view[1416], 189, "1416th value should be 189");
assertEq(view[1417], 76, "1417th value should be 76");
assertEq(view[1418], 15, "1418th value should be 15");
assertEq(view[1419], 255, "1419th value should be 255");
assertEq(view[1420], 135, "1420th value should be 135");
assertEq(view[1421], 217, "1421th value should be 217");
assertEq(view[1422], 17, "1422th value should be 17");
assertEq(view[1423], 255, "1423th value should be 255");
assertEq(view[1424], 82, "1424th value should be 82");
assertEq(view[1425], 102, "1425th value should be 102");
assertEq(view[1426], 20, "1426th value should be 20");
assertEq(view[1427], 255, "1427th value should be 255");
assertEq(view[1428], 43, "1428th value should be 43");
assertEq(view[1429], 166, "1429th value should be 166");
assertEq(view[1430], 58, "1430th value should be 58");
assertEq(view[1431], 255, "1431th value should be 255");
assertEq(view[1432], 56, "1432th value should be 56");
assertEq(view[1433], 230, "1433th value should be 230");
assertEq(view[1434], 45, "1434th value should be 45");
assertEq(view[1435], 255, "1435th value should be 255");
assertEq(view[1436], 0, "1436th value should be 0");
assertEq(view[1437], 0, "1437th value should be 0");
assertEq(view[1438], 0, "1438th value should be 0");
assertEq(view[1439], 255, "1439th value should be 255");
assertEq(view[1440], 0, "1440th value should be 0");
assertEq(view[1441], 0, "1441th value should be 0");
assertEq(view[1442], 0, "1442th value should be 0");
assertEq(view[1443], 255, "1443th value should be 255");
assertEq(view[1444], 0, "1444th value should be 0");
assertEq(view[1445], 0, "1445th value should be 0");
assertEq(view[1446], 0, "1446th value should be 0");
assertEq(view[1447], 255, "1447th value should be 255");
assertEq(view[1448], 0, "1448th value should be 0");
assertEq(view[1449], 0, "1449th value should be 0");
assertEq(view[1450], 0, "1450th value should be 0");
assertEq(view[1451], 255, "1451th value should be 255");
assertEq(view[1452], 0, "1452th value should be 0");
assertEq(view[1453], 0, "1453th value should be 0");
assertEq(view[1454], 0, "1454th value should be 0");
assertEq(view[1455], 255, "1455th value should be 255");
assertEq(view[1456], 0, "1456th value should be 0");
assertEq(view[1457], 0, "1457th value should be 0");
assertEq(view[1458], 0, "1458th value should be 0");
assertEq(view[1459], 255, "1459th value should be 255");
assertEq(view[1460], 0, "1460th value should be 0");
assertEq(view[1461], 0, "1461th value should be 0");
assertEq(view[1462], 0, "1462th value should be 0");
assertEq(view[1463], 255, "1463th value should be 255");
assertEq(view[1464], 21, "1464th value should be 21");
assertEq(view[1465], 153, "1465th value should be 153");
assertEq(view[1466], 132, "1466th value should be 132");
assertEq(view[1467], 255, "1467th value should be 255");
assertEq(view[1468], 176, "1468th value should be 176");
assertEq(view[1469], 12, "1469th value should be 12");
assertEq(view[1470], 28, "1470th value should be 28");
assertEq(view[1471], 255, "1471th value should be 255");
assertEq(view[1472], 82, "1472th value should be 82");
assertEq(view[1473], 102, "1473th value should be 102");
assertEq(view[1474], 20, "1474th value should be 20");
assertEq(view[1475], 255, "1475th value should be 255");
assertEq(view[1476], 189, "1476th value should be 189");
assertEq(view[1477], 76, "1477th value should be 76");
assertEq(view[1478], 15, "1478th value should be 15");
assertEq(view[1479], 255, "1479th value should be 255");
assertEq(view[1480], 189, "1480th value should be 189");
assertEq(view[1481], 76, "1481th value should be 76");
assertEq(view[1482], 15, "1482th value should be 15");
assertEq(view[1483], 255, "1483th value should be 255");
assertEq(view[1484], 243, "1484th value should be 243");
assertEq(view[1485], 191, "1485th value should be 191");
assertEq(view[1486], 12, "1486th value should be 12");
assertEq(view[1487], 255, "1487th value should be 255");
assertEq(view[1488], 243, "1488th value should be 243");
assertEq(view[1489], 191, "1489th value should be 191");
assertEq(view[1490], 12, "1490th value should be 12");
assertEq(view[1491], 255, "1491th value should be 255");
assertEq(view[1492], 41, "1492th value should be 41");
assertEq(view[1493], 51, "1493th value should be 51");
assertEq(view[1494], 10, "1494th value should be 10");
assertEq(view[1495], 255, "1495th value should be 255");
assertEq(view[1496], 41, "1496th value should be 41");
assertEq(view[1497], 51, "1497th value should be 51");
assertEq(view[1498], 10, "1498th value should be 10");
assertEq(view[1499], 255, "1499th value should be 255");
assertEq(view[1500], 41, "1500th value should be 41");
assertEq(view[1501], 51, "1501th value should be 51");
assertEq(view[1502], 10, "1502th value should be 10");
assertEq(view[1503], 255, "1503th value should be 255");
assertEq(view[1504], 41, "1504th value should be 41");
assertEq(view[1505], 51, "1505th value should be 51");
assertEq(view[1506], 10, "1506th value should be 10");
assertEq(view[1507], 255, "1507th value should be 255");
assertEq(view[1508], 41, "1508th value should be 41");
assertEq(view[1509], 51, "1509th value should be 51");
assertEq(view[1510], 10, "1510th value should be 10");
assertEq(view[1511], 255, "1511th value should be 255");
assertEq(view[1512], 41, "1512th value should be 41");
assertEq(view[1513], 51, "1513th value should be 51");
assertEq(view[1514], 10, "1514th value should be 10");
assertEq(view[1515], 255, "1515th value should be 255");
assertEq(view[1516], 41, "1516th value should be 41");
assertEq(view[1517], 51, "1517th value should be 51");
assertEq(view[1518], 10, "1518th value should be 10");
assertEq(view[1519], 255, "1519th value should be 255");
assertEq(view[1520], 94, "1520th value should be 94");
assertEq(view[1521], 166, "1521th value should be 166");
assertEq(view[1522], 7, "1522th value should be 7");
assertEq(view[1523], 255, "1523th value should be 255");
assertEq(view[1524], 94, "1524th value should be 94");
assertEq(view[1525], 166, "1525th value should be 166");
assertEq(view[1526], 7, "1526th value should be 7");
assertEq(view[1527], 255, "1527th value should be 255");
assertEq(view[1528], 94, "1528th value should be 94");
assertEq(view[1529], 166, "1529th value should be 166");
assertEq(view[1530], 7, "1530th value should be 7");
assertEq(view[1531], 255, "1531th value should be 255");
assertEq(view[1532], 94, "1532th value should be 94");
assertEq(view[1533], 166, "1533th value should be 166");
assertEq(view[1534], 7, "1534th value should be 7");
assertEq(view[1535], 255, "1535th value should be 255");
assertEq(view[1536], 148, "1536th value should be 148");
assertEq(view[1537], 25, "1537th value should be 25");
assertEq(view[1538], 5, "1538th value should be 5");
assertEq(view[1539], 255, "1539th value should be 255");
assertEq(view[1540], 148, "1540th value should be 148");
assertEq(view[1541], 25, "1541th value should be 25");
assertEq(view[1542], 5, "1542th value should be 5");
assertEq(view[1543], 255, "1543th value should be 255");
assertEq(view[1544], 148, "1544th value should be 148");
assertEq(view[1545], 25, "1545th value should be 25");
assertEq(view[1546], 5, "1546th value should be 5");
assertEq(view[1547], 255, "1547th value should be 255");
assertEq(view[1548], 148, "1548th value should be 148");
assertEq(view[1549], 25, "1549th value should be 25");
assertEq(view[1550], 5, "1550th value should be 5");
assertEq(view[1551], 255, "1551th value should be 255");
assertEq(view[1552], 148, "1552th value should be 148");
assertEq(view[1553], 25, "1553th value should be 25");
assertEq(view[1554], 5, "1554th value should be 5");
assertEq(view[1555], 255, "1555th value should be 255");
assertEq(view[1556], 148, "1556th value should be 148");
assertEq(view[1557], 25, "1557th value should be 25");
assertEq(view[1558], 5, "1558th value should be 5");
assertEq(view[1559], 255, "1559th value should be 255");
assertEq(view[1560], 148, "1560th value should be 148");
assertEq(view[1561], 25, "1561th value should be 25");
assertEq(view[1562], 5, "1562th value should be 5");
assertEq(view[1563], 255, "1563th value should be 255");
assertEq(view[1564], 148, "1564th value should be 148");
assertEq(view[1565], 25, "1565th value should be 25");
assertEq(view[1566], 5, "1566th value should be 5");
assertEq(view[1567], 255, "1567th value should be 255");
assertEq(view[1568], 148, "1568th value should be 148");
assertEq(view[1569], 25, "1569th value should be 25");
assertEq(view[1570], 5, "1570th value should be 5");
assertEq(view[1571], 255, "1571th value should be 255");
assertEq(view[1572], 148, "1572th value should be 148");
assertEq(view[1573], 25, "1573th value should be 25");
assertEq(view[1574], 5, "1574th value should be 5");
assertEq(view[1575], 255, "1575th value should be 255");
assertEq(view[1576], 202, "1576th value should be 202");
assertEq(view[1577], 140, "1577th value should be 140");
assertEq(view[1578], 2, "1578th value should be 2");
assertEq(view[1579], 255, "1579th value should be 255");
assertEq(view[1580], 202, "1580th value should be 202");
assertEq(view[1581], 140, "1581th value should be 140");
assertEq(view[1582], 2, "1582th value should be 2");
assertEq(view[1583], 255, "1583th value should be 255");
assertEq(view[1584], 202, "1584th value should be 202");
assertEq(view[1585], 140, "1585th value should be 140");
assertEq(view[1586], 2, "1586th value should be 2");
assertEq(view[1587], 255, "1587th value should be 255");
assertEq(view[1588], 202, "1588th value should be 202");
assertEq(view[1589], 140, "1589th value should be 140");
assertEq(view[1590], 2, "1590th value should be 2");
assertEq(view[1591], 255, "1591th value should be 255");
assertEq(view[1592], 202, "1592th value should be 202");
assertEq(view[1593], 140, "1593th value should be 140");
assertEq(view[1594], 2, "1594th value should be 2");
assertEq(view[1595], 255, "1595th value should be 255");
assertEq(view[1596], 202, "1596th value should be 202");
assertEq(view[1597], 140, "1597th value should be 140");
assertEq(view[1598], 2, "1598th value should be 2");
assertEq(view[1599], 255, "1599th value should be 255");

// Code used to generate the assertEq list above.
function generateAssertList() {
  function template(i, x) {
    return 'assertEq(view[' + i + '], ' + x + ', "' + i + 'th value should be ' + x + '");\n';
  }
  var buf = ''
  for (var i = 0; i < LIMIT_SHOW; i++)
      buf += template(i, view[i]);
  print(buf);
}
//generateAssertList();
