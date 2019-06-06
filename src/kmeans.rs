use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::float::Float;
use num_traits::identities::{One, Zero};
use rand::prelude::thread_rng;
use rand::prelude::IteratorRandom;
use rand::seq::index::sample;
use std::f32;
use std::iter::Sum;
use std::ops::AddAssign;

#[derive(Debug)]
pub struct Kmeans<T: Float + One + Zero + ScalarOperand + AddAssign + Copy + Sum> {
    pub centers: Vec<Array1<T>>,
    pub clusters: Vec<usize>,
    pub withinss: Vec<T>,
}

impl<T: Float + One + Zero + ScalarOperand + AddAssign + Copy + Sum> Kmeans<T> {
    pub fn new(data: &Array2<T>, nclust: usize, iterations: usize, nseeds: usize) -> Kmeans<T> {
        let mut rng = thread_rng();
        (0..nseeds)
            .map(|_| {
                let mut centers = data.outer_iter().choose_multiple(&mut rng, nclust).iter().map(|row| row.to_owned()).collect::<Vec<Array1<T>>>();
                let mut clusters = vec![0; data.rows()];
                let mut withinss = vec![T::zero(); nclust];
                for _ in 0..iterations {
                    let mut sums = vec![Array1::zeros((data.cols())); nclust];
                    let mut counts = vec![0; nclust];
                    withinss = vec![T::zero(); nclust];
                    for (row_idx, row) in data.outer_iter().enumerate() {
                        let (cluster, distance) = centers
                            .iter()
                            .enumerate()
                            .map(|(i, center)| (i, ((&row - center) * (&row - center)).sum()))
                            .map(|(i, x)| if x.is_nan() { (i, T::from(f32::MAX).expect("T::from(f32::MAX)")) } else { (i, x) })
                            .min_by(|(_, a), (_, b)| a.partial_cmp(&b).expect("PartialOrd distance from center"))
                            .expect("min distance from center");
                        clusters[row_idx] = cluster;
                        sums[cluster] = &sums[cluster] + &row;
                        counts[cluster] += 1;
                        withinss[cluster] += distance;
                    }
                    centers = sums
                        .into_iter()
                        .zip(counts.into_iter())
                        .map(|(sum, count)| sum / T::from(count).expect("T::from(usize)"))
                        .collect::<Vec<Array1<T>>>();
                }
                Kmeans {
                    centers: centers,
                    clusters: clusters,
                    withinss: withinss,
                }
            })
            .min_by(|a, b| a.withinss.iter().cloned().sum::<T>().partial_cmp(&b.withinss.iter().cloned().sum::<T>()).expect("withinss is not NAN"))
            .expect("min withinss")
    }

    pub fn predict(&self, data: &Array2<T>) -> Vec<usize> {
        data.outer_iter()
            .enumerate()
            .map(|(row_idx, row)| {
                self.centers
                    .iter()
                    .enumerate()
                    .map(|(i, center)| (i, ((&row - center) * (&row - center)).sum()))
                    .min_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap())
                    .unwrap()
                    .0
            })
            .collect::<Vec<usize>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                59.59375, 270.6875, 51.59375, 307.6875, 86.59375, 286.6875, 319.59375, 145.6875, 314.59375, 174.6875, 350.59375, 161.6875,
            ],
        )
        .unwrap();

        let model = Kmeans::new(&data, 2, 100, 10);
        assert!(model.centers.len() == 2);

        let classes = model.predict(&data);
        let class_a = classes[0];

        assert!(classes.iter().take(3).all(|x| *x == class_a));
        assert!(classes.iter().skip(3).all(|x| *x != class_a));
    }
}
