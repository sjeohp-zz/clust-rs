use crate::itertools::Itertools;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::prelude::*;
use num_traits::float::Float;
use num_traits::identities::{One, Zero};

#[derive(Debug)]
pub struct Dbscan<T: Float + One + Zero> {
    pub eps: T,
    pub min_points: usize,
    pub clusters: Vec<Option<usize>>,
}

impl<T: Float + One + Zero> Dbscan<T> {
    pub fn new(data: &Array2<T>, eps: T, min_points: usize, borders: bool) -> Dbscan<T> {
        let mut c = 0;
        let mut neighbours = Vec::with_capacity(data.rows());
        let mut sub_neighbours = Vec::with_capacity(data.rows());
        let mut visited = vec![false; data.rows()];
        let mut clusters = vec![None; data.rows()];
        let kdt = kdtree_init(&data);

        for (row_idx, row) in data.outer_iter().enumerate() {
            if !visited[row_idx] {
                visited[row_idx] = true;

                neighbours.clear();
                region_query(row.as_slice().unwrap(), eps, &kdt, &mut neighbours);
                neighbours.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                neighbours.dedup();

                if neighbours.len() >= min_points {
                    clusters[row_idx] = Some(c);
                    while let Some(neighbour_idx) = neighbours.pop() {
                        if borders {
                            clusters[neighbour_idx] = Some(c);
                        }
                        if !visited[neighbour_idx] {
                            visited[neighbour_idx] = true;
                            sub_neighbours.clear();
                            region_query(data.row(neighbour_idx).as_slice().unwrap(), eps, &kdt, &mut sub_neighbours);

                            if sub_neighbours.len() >= min_points {
                                if !borders {
                                    clusters[neighbour_idx] = Some(c);
                                }
                                neighbours.extend_from_slice(&sub_neighbours);
                                neighbours.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                                neighbours.dedup();
                            }
                        }
                    }
                    c += 1;
                }
            }
        }

        Dbscan {
            eps: eps,
            min_points: min_points,
            clusters: clusters,
        }
    }

    pub fn predict(&self, data: &Array2<T>, new_data: &Array2<T>) -> Vec<ClusterPrediction> {
        let mut neighbours = Vec::with_capacity(data.rows());
        let kdt = kdtree_init(&data);
        new_data
            .outer_iter()
            .map(|row| {
                neighbours.clear();
                region_query(row.as_slice().unwrap(), self.eps, &kdt, &mut neighbours);
                let mut neighbour_clusters = neighbours.iter().map(|idx| self.clusters[*idx]).unique().collect::<Vec<Option<usize>>>();
                if neighbours.len() >= self.min_points - 1 {
                    ClusterPrediction::Core(neighbour_clusters)
                } else if neighbour_clusters.iter().any(|c| c.is_some()) {
                    ClusterPrediction::Border(neighbour_clusters)
                } else {
                    ClusterPrediction::Noise
                }
            })
            .collect::<Vec<ClusterPrediction>>()
    }
}

fn kdtree_init<'a, T: Float + One + Zero>(data: &'a Array2<T>) -> KdTree<T, usize, &'a [T]> {
    let mut kdt = KdTree::new(data.cols());
    for (idx, row) in data.outer_iter().enumerate() {
        kdt.add(row.into_slice().unwrap(), idx);
    }
    kdt
}

fn region_query<'a, T: Float + One + Zero>(row: &'a [T], eps: T, kdt: &KdTree<T, usize, &'a [T]>, neighbours: &mut Vec<usize>) {
    for (_, neighbour_idx) in kdt.within(row, eps.powi(2), &squared_euclidean).expect("KdTree error checking point") {
        neighbours.push(*neighbour_idx);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClusterPrediction {
    Core(Vec<Option<usize>>),
    Border(Vec<Option<usize>>),
    Noise,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clusters() {
        let data = Array2::from_shape_vec((8, 2), vec![1.0, 2.0, 1.1, 2.2, 0.9, 1.9, 1.0, 2.1, -2.0, 3.0, -2.2, 3.1, -1.0, -2.0, -2.0, -1.0]).unwrap();
        let mut model = Dbscan::new(&data, 0.5, 2, false);
        let clustering = dbg!(model.clusters);
        assert!(clustering.iter().take(4).all(|x| *x == Some(0)));
        assert!(clustering.iter().skip(4).take(2).all(|x| *x == Some(1)));
        assert!(clustering.iter().skip(6).all(|x| *x == None));
    }

    #[test]
    fn test_border_points() {
        let data = Array2::from_shape_vec((5, 1), vec![1.55, 2.0, 2.1, 2.2, 2.65]).unwrap();

        let mut with = Dbscan::new(&data, 0.5, 3, true);
        let mut without = Dbscan::new(&data, 0.5, 3, false);
        let with_borders_clustering = dbg!(with.clusters);
        let without_borders_clustering = dbg!(without.clusters);
        assert!(with_borders_clustering.iter().all(|x| *x == Some(0)));
        assert!(without_borders_clustering.iter().take(1).all(|x| *x == None));
        assert!(without_borders_clustering.iter().skip(1).take(3).all(|x| *x == Some(0)));
        assert!(without_borders_clustering.iter().skip(4).all(|x| *x == None));
    }

    #[test]
    fn test_prediction() {
        let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 1.1, 2.2, 0.9, 1.9, 1.0, 2.1, -2.0, 3.0, -2.2, 3.1]).unwrap();
        let mut model = Dbscan::new(&data, 0.5, 2, false);

        let new_data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 4.0, 4.0]).unwrap();
        let classes = model.predict(&data, &new_data);

        if let ClusterPrediction::Core(c0) = classes.get(0).unwrap() {
            assert!(c0.iter().any(|c| *c == Some(0)));
        } else {
            panic!("{:?}", classes[0]);
        }
        assert!(classes[1] == ClusterPrediction::Noise);
    }
}
