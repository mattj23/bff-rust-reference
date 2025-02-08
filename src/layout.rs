use crate::{single_row_matrix, Point2, Point3, Result, SparseMat};
use faer::sparse::linalg::solvers::Lu;
use faer::Mat;
use faer::sparse::Triplet;

fn cumulative_sum(a: &[f64], scale: f64) -> Vec<f64> {
    let mut sum = 0.0;
    a.iter().map(|&x| {
        sum += x * scale;
        sum
    }).collect()
}

///
///
/// # Arguments
///
/// * `ub`: an array with a single row and `b_lengths.len()` columns
/// * `b_lengths`:
///
/// returns: Mat<f64, usize, usize>
fn calc_im_elen(
    ub: &Mat<f64>,
    b_lengths: &[f64],
) -> Mat<f64> {
    let ub_slice = ub.col_as_slice(0);
    let values = b_lengths.iter().enumerate()
        .map(|(i, &l)| {
            let a = (ub_slice[i] + ub_slice[(i + 1) % ub_slice.len()]) / 2.0;
            b_lengths[i] * a.exp()
        })
        .collect::<Vec<_>>();

    single_row_matrix(&values)
}

fn calc_best_fit_tangents(im_k: &[f64]) -> Mat<f64> {
    let phi = cumulative_sum(im_k, -1.0);
    Mat::from_fn(phi.len(), 2, |i, j| match j {
        0 => phi[i].cos(),
        1 => phi[i].sin(),
        _ => unreachable!()
    })
}

fn calc_boundary_vertex_masses(b_lengths: &[f64]) -> Vec<f64> {
    b_lengths.iter().enumerate()
        .map(|(i, &l)| (l + b_lengths[(i + 1) % b_lengths.len()]) / 2.0)
        .collect()
}

fn calc_best_fit_n1(b_lengths: &[f64]) -> Result<SparseMat> {
    let bvm = b_lengths.iter().enumerate()
        .map(|(i, &l)| (l + b_lengths[(i + 1) % b_lengths.len()]) / 2.0)
        .collect::<Vec<_>>();

    SparseMat::try_new_from_triplets(bvm.len(), bvm.len(),
                                     &bvm.iter().enumerate().map(|(i, &v)| Triplet::new(i as u32, i as u32, v)).collect::<Vec<_>>())
        .map_err(Into::into)
}

pub fn best_fit_curve(
    ub: &Mat<f64>,
    im_k: &[f64],
    i_bound: &[u32],
    b_lengths: &[f64],
) -> Result<Vec<f64>> {
    let phi = cumulative_sum(im_k, -1.0);
    let tangents = calc_best_fit_tangents(b_lengths);
    let im_elen = calc_im_elen(ub, b_lengths);
    let n1 = calc_best_fit_n1(b_lengths)?;

    let core = &tangents.transpose() * &n1 * &tangents;




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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use approx::assert_relative_eq;
    use crate::conformal::boundary_edge_lengths;
    use crate::invert_2x2;

    fn as_vec(mat: &Mat<f64>) -> Vec<Vec<f64>> {
        mat.row_iter().map(|r| r.iter().copied().collect()).collect()
    }

    fn mock_im_k() -> Vec<f64> {
        get_float_vector("dirichlet_im_k.floatvec")
    }

    #[test]
    fn bvm_calc() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let bvm = calc_boundary_vertex_masses(&b_lengths);

        let expected = get_float_vector("bestfitcurve_boundary_vertex_masses.floatvec");

        assert_vectors_eq!(bvm, expected);
        Ok(())
    }

    #[test]
    fn tangent_calc() -> Result<()> {
        let im_k = mock_im_k();
        let tangents = calc_best_fit_tangents(&im_k);

        let expected = get_float_matrix("bestfitcurve_tangents.floatmat");
        let check = as_vec(&tangents);

        assert_matrices_eq!(check, expected, 1e-8);

        Ok(())
    }

    #[test]
    fn bestfit_curve_n1_calc() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let n1 = calc_best_fit_n1(&b_lengths)?;

        let expected = get_sparse_triplets("bestfitcurve_N1.coo");

        let check = sparse_as_triplets(&n1);

        assert_triplets_eq!(check, expected);
        Ok(())
    }

    #[test]
    fn bestfit_curve_core_inv() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let im_k = mock_im_k();

        let tangents = calc_best_fit_tangents(&im_k);
        let n1 = calc_best_fit_n1(&b_lengths)?;

        // This will be a 2x2 matrix
        let core = &tangents.transpose() * &n1 * &tangents;
        let inv = invert_2x2(&core)?;

        let expected = get_float_matrix("bestfitcurve_inv_stuff.floatmat");
        let check = as_vec(&inv);

        assert_matrices_eq!(check, expected, 1e-5);

        Ok(())
    }

    #[test]
    fn im_elen_test() -> Result<()> {
        let mesh = get_test_structure();
        let b_len = boundary_edge_lengths(&mesh)?;

        let ub = Mat::<f64>::zeros(b_len.len(), 1);
        let im_elen = calc_im_elen(&ub, &b_len);

        let expected = get_float_vector("bestfitcurve_im_elen.floatvec");
        let check = im_elen.row(0).iter().copied().collect::<Vec<_>>();

        assert_vectors_eq!(check, expected);
        Ok(())
    }

}