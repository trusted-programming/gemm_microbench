#![allow(non_snake_case)]
extern crate intel_mkl_src;

use matrixmultiply::dgemm as dgemm_matrixmultiply;
use cblas::{dgemm as dgemm_cblas, Layout, Transpose};
use core::slice::{from_raw_parts, from_raw_parts_mut};

#[no_mangle]
pub extern "C" fn matmul_matrixmultiply(
    out: *mut f64,
    inp: *const f64,
    weight: *const f64,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    unsafe {
        dgemm_matrixmultiply(
            B * T, // m
            C, // k
            OC, // n
            1.0,
            inp,
            C as isize, // a_row_stride
            1,                // a_col_stride
            weight,
            OC as isize, // b_row_stride
            1,                 // b_col_stride
            0.0,
            out,
            OC as isize, // c_row_stride
            1,                 // c_col_stride
        );
    };
}

#[no_mangle]
pub extern "C" fn matmul_cblas(
    out: *mut f64,
    inp: *const f64,
    weight: *const f64,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    unsafe {
        let m = B * T;
        let k = C;
        let n = OC;

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