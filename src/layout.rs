use crate::{Point2, Point3, Result, SparseMat};
use faer::sparse::linalg::solvers::Lu;
use faer::Mat;

pub fn best_fit_curve(
    ub: &Mat<f64>,
    im_k: &[f64],
    i_bound: &[u32],
    b_lengths: &[f64],
) -> Result<Vec<f64>> {
    // Things it needs:
    //

    todo!("Implement best_fit_curve")
}

pub fn extend_curve(
    a_lu: &Lu<u32, f64>,
    aii_lu: &Lu<u32, f64>,
    aib: &SparseMat,
    vertices: &[Point3],
    i_bound: &[u32],
    i_inner: &[u32],
) -> Result<Vec<Point2>> {
    todo!("Implement extend_curve")
}
