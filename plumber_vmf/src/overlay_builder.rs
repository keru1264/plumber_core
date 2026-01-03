use crate::{
    builder_utils::{built_polygon_center, polygon_center, BuiltPolygon},
    vmf_entity::Entity,
};
use glam::{Mat3, Vec2, Vec3};
use itertools::Itertools;

/// Builds overlay data from a VMF entity
pub struct OverlayBuilder<'a> {
    entity: &'a Entity,
    basis_origin: Vec3,
    basis_u: Vec3,
    basis_v: Vec3,
    basis_normal: Vec3,
    start_u: f32,
    end_u: f32,
    start_v: f32,
    end_v: f32,
    uv_points: [Vec3; 4],
}

impl<'a> OverlayBuilder<'a> {
    pub fn new(entity: &'a Entity) -> Option<Self> {
        let basis_origin = entity.property("BasisOrigin")?.parse().ok()?;
        let basis_u = entity.property("BasisU")?.parse().ok()?;
        let basis_v = entity.property("BasisV")?.parse().ok()?;
        let basis_normal = entity.property("BasisNormal")?.parse().ok()?;

        let start_u: f32 = entity.property("StartU")?.parse().ok()?;
        let end_u: f32 = entity.property("EndU")?.parse().ok()?;
        let start_v: f32 = entity.property("StartV")?.parse().ok()?;
        let end_v: f32 = entity.property("EndV")?.parse().ok()?;

        let uv0 = entity.property("uv0")?.parse().ok()?;
        let uv1 = entity.property("uv1")?.parse().ok()?;
        let uv2 = entity.property("uv2")?.parse().ok()?;
        let uv3 = entity.property("uv3")?.parse().ok()?;

        Some(Self {
            entity,
            basis_origin,
            basis_u,
            basis_v,
            basis_normal,
            start_u,
            end_u,
            start_v,
            end_v,
            uv_points: [uv0, uv1, uv2, uv3],
        })
    }

    pub fn material(&self) -> Option<&str> {
        self.entity.property("material")
    }

    pub fn sides(&self) -> impl Iterator<Item = i32> + '_ {
        self.entity
            .property("sides")
            .into_iter()
            .flat_map(|s| s.split_whitespace())
            .filter_map(|s| s.parse().ok())
    }

    /// Cuts the overlay polygon from the provided face polygons.
    /// Returns vertices and UVs for the cut overlay geometry.
    pub fn cut_faces(&self, faces: &[BuiltPolygon]) -> Option<(Vec<Vec3>, Vec<Vec2>)> {
        if faces.is_empty() {
            return None;
        }

        // Build the transformation matrix from world space to UV space
        let basis_matrix = Mat3::from_cols(self.basis_u, self.basis_v, self.basis_normal);
        let inv_basis = basis_matrix.transpose();

        // Calculate the overlay polygon in UV space
        let overlay_poly_uv: Vec<Vec2> = self
            .uv_points
            .iter()
            .map(|p| {
                let local = inv_basis * (*p - self.basis_origin);
                Vec2::new(local.x, local.y)
            })
            .collect();

        // Compute the actual center of all face geometry
        let face_center = if faces.len() == 1 {
            built_polygon_center(&faces[0])
        } else {
            // Average of all face centers
            let sum: Vec3 = faces.iter().map(|f| built_polygon_center(f)).sum();
            sum / faces.len() as f32
        };

        // Compute offset between BasisOrigin and actual face geometry center.
        // This fixes the issue where BSPSource moves BasisOrigin to displacement
        // center but the overlay faces are elsewhere.
        let origin_offset = face_center - self.basis_origin;
        let origin_offset_uv = inv_basis * origin_offset;
        let offset_2d = Vec2::new(origin_offset_uv.x, origin_offset_uv.y);

        let mut all_vertices = Vec::new();
        let mut all_uvs = Vec::new();

        for face in faces {
            // Transform face vertices to UV space, applying the offset correction
            let uv_space_vertices: Vec<Vec2> = face
                .iter()
                .map(|(pos, _)| {
                    // Transform relative to BasisOrigin but account for the offset
                    let local = inv_basis * (*pos - self.basis_origin);
                    Vec2::new(local.x, local.y) - offset_2d
                })
                .collect();

            // Clip the face polygon against the overlay polygon
            let clipped = clip_polygon(&uv_space_vertices, &overlay_poly_uv);

            if clipped.len() < 3 {
                continue;
            }

            // Transform back to world space and compute UVs
            for uv_pos in &clipped {
                // Add offset back when transforming to world space
                let adjusted_uv = *uv_pos + offset_2d;
                let world_pos = self.basis_origin
                    + self.basis_u * adjusted_uv.x
                    + self.basis_v * adjusted_uv.y;
                all_vertices.push(world_pos);

                // Compute texture UVs based on position within overlay bounds
                let u = remap(uv_pos.x, self.uv_points[0].x, self.uv_points[2].x, self.start_u, self.end_u);
                let v = remap(uv_pos.y, self.uv_points[0].y, self.uv_points[2].y, self.start_v, self.end_v);
                all_uvs.push(Vec2::new(u, v));
            }
        }

        if all_vertices.is_empty() {
            None
        } else {
            Some((all_vertices, all_uvs))
        }
    }
}

/// Remaps a value from one range to another
fn remap(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let from_range = from_max - from_min;
    if from_range.abs() < f32::EPSILON {
        return to_min;
    }
    let t = (value - from_min) / from_range;
    to_min + t * (to_max - to_min)
}

/// Clips a polygon against another polygon using Sutherland-Hodgman algorithm
fn clip_polygon(subject: &[Vec2], clip: &[Vec2]) -> Vec<Vec2> {
    if subject.is_empty() || clip.len() < 3 {
        return Vec::new();
    }

    let mut output = subject.to_vec();

    for (i, &clip_v1) in clip.iter().enumerate() {
        if output.is_empty() {
            break;
        }

        let clip_v2 = clip[(i + 1) % clip.len()];
        let edge = clip_v2 - clip_v1;
        let edge_normal = Vec2::new(-edge.y, edge.x);

        let input = output;
        output = Vec::new();

        for (j, &v1) in input.iter().enumerate() {
            let v2 = input[(j + 1) % input.len()];

            let d1 = (v1 - clip_v1).dot(edge_normal);
            let d2 = (v2 - clip_v1).dot(edge_normal);

            let v1_inside = d1 >= 0.0;
            let v2_inside = d2 >= 0.0;

            if v1_inside {
                output.push(v1);
                if !v2_inside {
                    // Exiting: add intersection
                    if let Some(intersection) = line_intersection(v1, v2, clip_v1, clip_v2) {
                        output.push(intersection);
                    }
                }
            } else if v2_inside {
                // Entering: add intersection
                if let Some(intersection) = line_intersection(v1, v2, clip_v1, clip_v2) {
                    output.push(intersection);
                }
            }
        }
    }

    output
}

/// Computes the intersection point of two line segments
fn line_intersection(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> Option<Vec2> {
    let d1 = a2 - a1;
    let d2 = b2 - b1;

    let cross = d1.x * d2.y - d1.y * d2.x;

    if cross.abs() < f32::EPSILON {
        return None;
    }

    let d = b1 - a1;
    let t = (d.x * d2.y - d.y * d2.x) / cross;

    Some(a1 + d1 * t)
}

/// Computes the center of a built polygon
fn built_polygon_center(polygon: &BuiltPolygon) -> Vec3 {
    if polygon.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = polygon.iter().map(|(pos, _)| *pos).sum();
    sum / polygon.len() as f32
}
