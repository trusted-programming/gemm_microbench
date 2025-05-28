#![allow(non_snake_case)]
extern crate intel_mkl_src;

use matrixmultiply::dgemm as dgemm_matrixmultiply;
use cblas::{dgemm as dgemm_cblas, Layout, Transpose};
use core::slice::{from_raw_parts, from_raw_parts_mut};

// Define the matrix dimensions as constants
const B_CONST: usize = 1;
const T_CONST: usize = 5000;
const C_CONST: usize = 5000;
const OC_CONST: usize = 5000;

#[no_mangle]
pub extern "C" fn matmul_matrixmultiply(
    out: *mut f64,
    inp: *const f64,
    weight: *const f64,
    _B: usize,
    _T: usize,
    _C: usize,
    _OC: usize,
) {
    unsafe {
        dgemm_matrixmultiply(
            B_CONST * T_CONST, // m
            C_CONST, // k
            OC_CONST, // n
            1.0,
            inp,
            C_CONST as isize, // a_row_stride
            1,                // a_col_stride
            weight,
            OC_CONST as isize, // b_row_stride
            1,                 // b_col_stride
            0.0,
            out,
            OC_CONST as isize, // c_row_stride
            1,                 // c_col_stride
        );
    };
}

#[no_mangle]
pub extern "C" fn matmul_cblas(
    out: *mut f64,
    inp: *const f64,
    weight: *const f64,
    _B: usize,
    _T: usize,
    _C: usize,
    _OC: usize,
) {
    unsafe {
        let m = B_CONST * T_CONST;
        let k = C_CONST;
        let n = OC_CONST;

        let out_slice = from_raw_parts_mut(out, m * n);
        let inp_slice = from_raw_parts(inp, m * k);
        let weight_slice = from_raw_parts(weight, k * n);

        dgemm_cblas(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            inp_slice,
            k as i32,
            weight_slice,
            n as i32,
            0.0,
            out_slice,
            n as i32,
        );
    }
}