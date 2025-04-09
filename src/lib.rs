#![allow(non_snake_case)]
use matrixmultiply::sgemm;

#[no_mangle]
pub extern "C" fn matmul(
    out: *mut f32,
    inp: *const f32,
    weight: *const f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    unsafe { 
        sgemm(
            B * T,
            C,
            OC,
            1.0,
            inp,
            C as isize,
            1,
            weight,
            1,
            C as isize,
            0.0,
            out,
            OC as isize,
            1,
        );
    };
}