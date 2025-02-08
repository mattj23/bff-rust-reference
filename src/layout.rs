use crate::{invert_2x2, single_col_matrix, Result, SparseMat};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::Triplet;
use faer::Mat;

fn cumulative_sum(a: &[f64], scale: f64) -> Vec<f64> {
    let mut sum = 0.0;
    a.iter()
        .map(|&x| {
            sum += x * scale;
            sum
        })
        .collect()
}

fn zip_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

///
///
/// # Arguments
///
/// * `ub`: an array with a single row and `b_lengths.len()` columns
/// * `b_lengths`:
///
/// returns: Mat<f64, usize, usize>
fn calc_im_elen(ub: &Mat<f64>, b_lengths: &[f64]) -> Mat<f64> {
    let ub_slice = ub.col_as_slice(0);
    let values = b_lengths
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let a = (ub_slice[i] + ub_slice[(i + 1) % ub_slice.len()]) / 2.0;
            l * a.exp()
        })
        .collect::<Vec<_>>();

    single_col_matrix(&values)
}

fn calc_best_fit_tangents(im_k: &[f64]) -> Mat<f64> {
    let phi = cumulative_sum(im_k, -1.0);
    Mat::from_fn(phi.len(), 2, |i, j| match j {
        0 => phi[i].cos(),
        1 => phi[i].sin(),
        _ => unreachable!(),
    })
}

fn calc_boundary_vertex_masses(b_lengths: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; b_lengths.len()];
    for (i, l) in b_lengths.iter().enumerate() {
        let ni = (i + 1) % b_lengths.len();
        result[ni] = (l + b_lengths[ni]) / 2.0;
    }
    result
}

fn calc_best_fit_n1(b_lengths: &[f64]) -> Result<SparseMat> {
    let bvm = calc_boundary_vertex_masses(b_lengths);
    SparseMat::try_new_from_triplets(
        bvm.len(),
        bvm.len(),
        &bvm.iter()
            .enumerate()
            .map(|(i, &v)| Triplet::new(i as u32, i as u32, v))
            .collect::<Vec<_>>(),
    )
    .map_err(Into::into)
}

fn modified_im_elen(im_elen: &Mat<f64>, n1: &SparseMat, tangents: &Mat<f64>) -> Mat<f64> {
    let core = invert_2x2(&(&tangents.transpose() * n1 * tangents)).unwrap();
    let im_elen_sub = n1 * tangents * core * tangents.transpose() * im_elen;
    im_elen - im_elen_sub
}

pub fn best_fit_curve(ub: &Mat<f64>, im_k: &[f64], b_lengths: &[f64]) -> Result<Mat<f64>> {
    let tangents = calc_best_fit_tangents(im_k);
    let im_elen = calc_im_elen(ub, b_lengths);
    let n1 = calc_best_fit_n1(b_lengths)?;

    // Modify the im_elen matrix
    let im_elen = modified_im_elen(&im_elen, &n1, &tangents);

    // Any negative values in im_elen indicate that there is a boundary edge with a
    // negative length.
    if im_elen.col_as_slice(0).iter().any(|&x| x < 0.0) {
        return Err("Negative values in im_elen".into());
    }

    // We'll do a row-wise multiplication of im_elen against the tangents matrix, so that the
    // first column of the result is im_elen[(i, 0)] * tangents[(i, 0)] and the second is
    // im_elen[(i, 0)] * tangents[(i, 1)]
    let col0 = zip_product(im_elen.col_as_slice(0), tangents.col_as_slice(0));
    let col1 = zip_product(im_elen.col_as_slice(0), tangents.col_as_slice(1));

    // We'll take the cumulative sum of each
    let col0 = cumulative_sum(&col0, 1.0);
    let col1 = cumulative_sum(&col1, 1.0);

    // Finally, the result will be combined into a 2-column matrix and rolled forward by one
    Ok(Mat::from_fn(col0.len(), 2, |i, j| {
        let insert_i = ((i + col0.len()) - 1) % col0.len();
        match j {
            0 => col0[insert_i],
            1 => col1[insert_i],
            _ => unreachable!(),
        }
    }))
}

fn calc_extend_h(n_vert: usize, i_bound: &[u32], uvb: &Mat<f64>) -> Mat<f64> {
    let mut h = Mat::zeros(n_vert, 1);
    let bn = uvb.shape().0;
    for (i_b, &i_all) in i_bound.iter().enumerate() {
        // The value `i_b` is the index of the boundary vertex in the `i_bound` array, and `i_all`
        // is the index of the same boundary vertex in the `vertices` array.
        let i_b_prev = (i_b + bn - 1) % bn;
        let i_b_next = (i_b + 1) % bn;
        h[(i_all as usize, 0)] = 0.5 * (uvb[(i_b_prev, 0)] - uvb[(i_b_next, 0)]);
    }

    h
}

pub fn calc_extend_uv_xs(
    n_vert: usize,
    i_bound: &[u32],
    i_inner: &[u32],
    uvb: &Mat<f64>,
    aib: &SparseMat,
    aii_lu: &Lu<u32, f64>,
) -> Mat<f64> {
    let mut uv = Mat::zeros(n_vert, 2);

    // Copy the boundary vertex x values into the first column of the uv matrix from the ubv matrix
    for (&i_b, &v) in i_bound.iter().zip(uvb.col_as_slice(0)) {
        uv[(i_b as usize, 0)] = v;
    }

    let core = aib * &uvb.subcols(0, 1);
    let uv_inner = aii_lu.solve(-core);

    for (&i, &v) in i_inner.iter().zip(uv_inner.col_as_slice(0)) {
        uv[(i as usize, 0)] = v;
    }

    uv
}

pub fn extend_curve(
    a_lu: &Lu<u32, f64>,
    aii_lu: &Lu<u32, f64>,
    aib: &SparseMat,
    uvb: &Mat<f64>,
    n_vert: usize,
    i_bound: &[u32],
    i_inner: &[u32],
) -> Result<Vec<[f64; 2]>> {
    // Calculate the x values for all vertices
    let uv = calc_extend_uv_xs(n_vert, i_bound, i_inner, uvb, aib, aii_lu);

    // Solve for the y values
    let h = calc_extend_h(n_vert, i_bound, uvb);
    let y = a_lu.solve(-&h);

    Ok(uv
        .col_as_slice(0)
        .iter()
        .zip(y.col_as_slice(0).iter())
        .map(|(&x, &y)| [x, y])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::boundary_edge_lengths;
    use crate::conformal::tests::laplacian_mock;
    use crate::invert_2x2;
    use crate::test_utils::*;
    use approx::assert_relative_eq;

    #[test]
    fn extend_curve_uv_xs() -> Result<()> {
        let mesh = get_test_structure();
        let i_bound = mesh.single_boundary_vertices()?;
        let (_, aii, aib, _) = laplacian_mock()?;
        let aii_lu = aii.sp_lu()?;

        let uv_xs = calc_extend_uv_xs(
            mesh.vertices.len(),
            &i_bound,
            &mesh.inner_vertices()?,
            &mock_uvb(),
            &aib,
            &aii_lu,
        );

        let expected = get_float_matrix("extendcurve_uv_pre_extension.floatmat");
        let check = as_vec(&uv_xs);

        assert_matrices_eq!(check, expected, 1e-8);

        Ok(())
    }

    #[test]
    fn extend_curve_h() -> Result<()> {
        let mesh = get_test_structure();
        let i_bound = mesh.single_boundary_vertices()?;
        let uvb = mock_uvb();

        let h = calc_extend_h(mesh.vertices.len(), &i_bound, &uvb);

        let expected = get_float_vector("extendcurve_h.floatvec");
        let check = h.col_as_slice(0).to_vec();

        assert_vectors_eq!(check, expected);

        Ok(())
    }

    #[test]
    fn bestfit_curve_full_calc() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let im_k = mock_im_k();
        let ub = Mat::<f64>::zeros(b_lengths.len(), 1);

        let result = best_fit_curve(&ub, &im_k, &b_lengths)?;

        let expected = get_float_matrix("extendcurve_uvb.floatmat");
        let check = as_vec(&result);

        assert_matrices_eq!(check, expected, 1e-8);
        Ok(())
    }

    #[test]
    fn bestfit_csum_calc() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let im_k = mock_im_k();

        let ub = Mat::<f64>::zeros(b_lengths.len(), 1);
        let im_elen = calc_im_elen(&ub, &b_lengths);

        let tangents = calc_best_fit_tangents(&im_k);
        let n1 = calc_best_fit_n1(&b_lengths)?;
        let im_elen = modified_im_elen(&im_elen, &n1, &tangents);

        // We'll do a row-wise multiplication of im_elen against the tangents matrix, so that the
        // first column of the result is im_elen[(i, 0)] * tangents[(i, 0)] and the second is
        // im_elen[(i, 0)] * tangents[(i, 1)]
        let col0 = zip_product(im_elen.col_as_slice(0), tangents.col_as_slice(0));
        let col1 = zip_product(im_elen.col_as_slice(0), tangents.col_as_slice(1));

        // We'll take the cumulative sum of each
        let col0 = cumulative_sum(&col0, 1.0);
        let col1 = cumulative_sum(&col1, 1.0);

        let check = col0
            .iter()
            .zip(col1.iter())
            .map(|(&a, &b)| vec![a, b])
            .collect::<Vec<_>>();
        let expected = get_float_matrix("bestfitcurve_csum.floatmat");

        assert_matrices_eq!(check, expected, 1e-8);

        Ok(())
    }

    #[test]
    fn bestfit_curve_im_elen_mod() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let im_k = mock_im_k();

        let ub = Mat::<f64>::zeros(b_lengths.len(), 1);
        let im_elen = calc_im_elen(&ub, &b_lengths);

        let tangents = calc_best_fit_tangents(&im_k);
        let n1 = calc_best_fit_n1(&b_lengths)?;
        let im_elen = modified_im_elen(&im_elen, &n1, &tangents);

        let expected = get_float_vector("bestfitcurve_im_elen_mod.floatvec");
        let check = im_elen.col_as_slice(0).to_vec();

        assert_vectors_eq!(check, expected);
        Ok(())
    }

    #[test]
    fn boundary_edge_lengths_check() -> Result<()> {
        let mesh = get_test_structure();
        let b_lengths = boundary_edge_lengths(&mesh)?;
        let expected = get_float_vector("boundary_edge_lengths.floatvec");

        assert_vectors_eq!(b_lengths, expected);
        Ok(())
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

        assert_matrices_eq!(check, expected, 1e-8);

        Ok(())
    }

    #[test]
    fn im_elen_test() -> Result<()> {
        let mesh = get_test_structure();
        let b_len = boundary_edge_lengths(&mesh)?;

        let ub = Mat::<f64>::zeros(b_len.len(), 1);
        let im_elen = calc_im_elen(&ub, &b_len);

        let expected = get_float_vector("bestfitcurve_im_elen.floatvec");
        let check = im_elen.col_as_slice(0).to_vec();

        assert_vectors_eq!(check, expected);
        Ok(())
    }

    fn as_vec(mat: &Mat<f64>) -> Vec<Vec<f64>> {
        mat.row_iter()
            .map(|r| r.iter().copied().collect())
            .collect()
    }

    fn mock_im_k() -> Vec<f64> {
        get_float_vector("dirichlet_im_k.floatvec")
    }

    fn mock_uvb() -> Mat<f64> {
        let data = get_float_matrix("extendcurve_uvb.floatmat");
        Mat::from_fn(data.len(), data[0].len(), |i, j| data[i][j])
    }
}
