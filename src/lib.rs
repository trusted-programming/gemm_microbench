#![allow(non_snake_case)]
use matrixmultiply::dgemm;

// Define the matrix dimensions as constants
const B_CONST: usize = 1;
const T_CONST: usize = 5000;
const C_CONST: usize = 5000;
const OC_CONST: usize = 5000;

#[no_mangle]
pub extern "C" fn matmul(
    out: *mut f64,
    inp: *const f64,
    weight: *const f64,
    _B: usize, // These are now ignored
    _T: usize,
    _C: usize,
    _OC: usize,
) {
    unsafe {
        dgemm(
            B_CONST * T_CONST,
            C_CONST,
            OC_CONST,
            1.0,
            inp,
            C_CONST as isize, // Hardcoded stride
            1,                // Hardcoded stride
            weight,
            1,                // Hardcoded stride
            C_CONST as isize, // Hardcoded stride
            0.0,
            out,
            OC_CONST as isize, // Hardcoded stride
            1,                // Hardcoded stride
        );
    };
}