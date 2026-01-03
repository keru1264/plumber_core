use std::collections::BTreeMap;

use glam::{Vec2, Vec3};
use itertools::Itertools;
use plumber_vmt::MaterialInfo;

use crate::{
    builder_utils::{
        building_polygons_to_face_mesh, polygon_center, GeometrySettings, PolygonWithNormals,
    },
    vmf_entity::{Entity, Overlay, Side, Solid},
};

#[derive(Debug, Clone)]
pub struct BuiltOverlay {
    pub id: i32,
    pub material_name: String,
    pub position: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub mesh: OverlayMesh,
}

#[derive(Debug, Clone, Default)]
pub struct OverlayMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub alpha: Vec<f32>,
}

impl OverlayMesh {
    #[must_use]
    pub fn indices(&self) -> Vec<u32> {
        (0..self.positions.len().saturating_sub(2))
            .flat_map(|i| [0, i as u32 + 1, i as u32 + 2])
            .collect()
    }

    fn recenter(&mut self) -> Vec3 {
        let center = polygon_center(&self.positions);
        for pos in &mut self.positions {
            *pos -= center;
        }
        center
    }
}

pub struct OverlayBuilder<'a> {
    overlay: &'a Overlay,
    settings: &'a GeometrySettings,
    origin: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    uv_data: UvData,
    // side_faces: Option<SideFaces<'a>>,
    sides_meshes: BTreeMap<i32, PolygonWithNormals>,
    render_order: i32,
}

#[derive(Debug, Clone)]
struct UvData {
    pub start_u: f32,
    pub end_u: f32,
    pub start_v: f32,
    pub end_v: f32,
}

impl<'a> OverlayBuilder<'a> {
    #[must_use]
    pub fn new(
        overlay: &'a Overlay,
        settings: &'a GeometrySettings,
        sides_meshes: BTreeMap<i32, PolygonWithNormals>,
    ) -> Self {
        Self {
            overlay,
            settings,
            origin: overlay.origin,
            u_axis: overlay.basis_u,
            v_axis: overlay.basis_v,
            uv_data: UvData {
                start_u: overlay.start_u,
                end_u: overlay.end_u,
                start_v: overlay.start_v,
                end_v: overlay.end_v,
            },
            sides_meshes,
            render_order: overlay.render_order.unwrap_or(0),
        }
    }

    /// Projects the overlay uv points to world coordinates.
    fn project_uv_to_world(&self, uv: Vec2) -> Vec3 {
        self.origin + self.u_axis * uv.x + self.v_axis * uv.y
    }

    /// Projects a world position to the overlay plane uv coordinates.
    fn project_world_to_uv(&self, pos: Vec3) -> Vec2 {
        let offset = pos - self.origin;
        let u = offset.dot(self.u_axis.normalize()) / self.u_axis.length();
        let v = offset.dot(self.v_axis.normalize()) / self.v_axis.length();
        Vec2::new(u, v)
    }

    /// Projects a world position to the overlay plane.
    fn project_onto_overlay_plane(&self, pos: Vec3) -> Vec3 {
        let uv = self.project_world_to_uv(pos);
        self.project_uv_to_world(uv)
    }

    /// Returns the normals of the overlay pointing perpendicular to u and v axis.
    fn overlay_plane_normals(&self) -> [Vec3; 2] {
        let n1 = self.u_axis.cross(self.v_axis).normalize();
        let n2 = -n1;
        [n1, n2]
    }

    fn compute_mesh_uv(&self, pos: Vec3) -> Vec2 {
        let uv = self.project_world_to_uv(pos);
        // remap uv from (uv_data.start_u, uv_data.end_u) to (0, 1)
        let u = (uv.x - self.uv_data.start_u) / (self.uv_data.end_u - self.uv_data.start_u);
        // flip v
        let v = 1.0 - (uv.y - self.uv_data.start_v) / (self.uv_data.end_v - self.uv_data.start_v);
        Vec2::new(u, v)
    }

    fn uv_polygon(&self) -> [Vec2; 4] {
        let uv_points = &self.overlay.uv_points;
        // the 4 inner arrays are the 4 corners of the overlay
        // the first point of each array is the corner
        // we want to get the first point of each array
        [uv_points[0][0], uv_points[1][0], uv_points[2][0], uv_points[3][0]]
    }

    #[must_use]
    pub fn build_mesh(self, _material: Option<&MaterialInfo>) -> Option<BuiltOverlay> {
        // get the 4 points of the overlay in world coordinates
        let uv_polygon = self.uv_polygon();
        let world_polygon = uv_polygon.map(|uv| self.project_uv_to_world(uv));

        let mut builder = OverlayMesh::default();
        let normals = self.overlay_plane_normals();

        for (side_id, side_mesh) in &self.sides_meshes {
            let side_center = polygon_center(&side_mesh.vertices);
            let side_center_proj = self.project_onto_overlay_plane(side_center);

            // test if the side center is inside the overlay
            // let inside = point_in_polygon(side_center_proj, &world_polygon);
            let closest = crate::builder_utils::closest_edge_point(&side_center_proj, &world_polygon);
            let inside = (side_center_proj - closest).length_squared() < 1e-6 
                || is_inside_convex_polygon(&side_center_proj, &world_polygon);

            // pick the normal that points to the same general direction as the face normals
            let face_normal = side_mesh.normals[0];
            let normal_facing_face = normals.iter().max_by(|a, b| {
                a.dot(face_normal).partial_cmp(&b.dot(face_normal)).unwrap()
            }).copied().unwrap_or(normals[0]);

            if inside {
                // clip side polygon to overlay polygon
                let mut clipped = side_mesh.vertices.clone();
                let mut clipped_normals = side_mesh.normals.clone();
                clip_polygon_with_normals(&mut clipped, &mut clipped_normals, &world_polygon);
                if clipped.len() < 3 {
                    continue;
                }

                // offset clipped vertices slightly along the normal
                let offset = normal_facing_face * (0.1 + self.render_order as f32 * 0.01);
                for vertex in &mut clipped {
                    *vertex += offset;
                }

                // add the polygon to the mesh
                let start_idx = builder.positions.len();
                builder.positions.extend(clipped.iter().copied());
                builder.normals.extend(clipped_normals);
                builder.uvs.extend(clipped.iter().map(|pos| self.compute_mesh_uv(*pos)));
                builder.alpha.extend(std::iter::repeat(1.0).take(clipped.len()));
            }
        }

        if builder.positions.is_empty() {
            return None;
        }

        // builder.recenter();
        //builder.recenter();

        Some(BuiltOverlay {
            id: self.overlay.id,
            material_name: self.overlay.material.clone(),
            position: Vec3::ZERO,
            u: self.u_axis,
            v: self.v_axis,
            mesh: builder,
        })
    }
}

/// Tests if a point is inside a convex polygon.
fn is_inside_convex_polygon(point: &Vec3, polygon: &[Vec3]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut sign = None;
    for i in 0..n {
        let v0 = polygon[i];
        let v1 = polygon[(i + 1) % n];
        let edge = v1 - v0;
        let to_point = *point - v0;
        let cross = edge.cross(to_point);
        let dot = cross.dot(cross);
        if dot.abs() < 1e-10 {
            continue;
        }
        let current_sign = dot > 0.0;
        match sign {
            None => sign = Some(current_sign),
            Some(s) if s != current_sign => return false,
            _ => {}
        }
    }
    true
}

/// Clips a polygon with vertex normals against another convex polygon.
fn clip_polygon_with_normals(
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    clip_polygon: &[Vec3],
) {
    let n = clip_polygon.len();
    for i in 0..n {
        let edge_start = clip_polygon[i];
        let edge_end = clip_polygon[(i + 1) % n];
        let edge_dir = edge_end - edge_start;

        // compute the inward normal of the clipping edge
        // assuming the polygon is wound counter-clockwise
        let center = polygon_center(clip_polygon);
        let to_center = center - edge_start;
        let edge_normal_candidate = edge_dir.cross(to_center.cross(edge_dir)).normalize();
        let edge_normal = if edge_normal_candidate.dot(to_center) > 0.0 {
            edge_normal_candidate
        } else {
            -edge_normal_candidate
        };

        // clip the polygon against the edge
        let mut new_vertices = Vec::with_capacity(vertices.len());
        let mut new_normals = Vec::with_capacity(normals.len());
        for j in 0..vertices.len() {
            let v0 = vertices[j];
            let v1 = vertices[(j + 1) % vertices.len()];
            let n0 = normals[j];
            let n1 = normals[(j + 1) % normals.len()];
            let d0 = (v0 - edge_start).dot(edge_normal);
            let d1 = (v1 - edge_start).dot(edge_normal);

            if d0 >= 0.0 {
                // v0 is inside
                new_vertices.push(v0);
                new_normals.push(n0);
                if d1 < 0.0 {
                    // v1 is outside, compute intersection
                    let t = d0 / (d0 - d1);
                    let intersection = v0 + t * (v1 - v0);
                    let normal = n0.lerp(n1, t).normalize();
                    new_vertices.push(intersection);
                    new_normals.push(normal);
                }
            } else if d1 >= 0.0 {
                // v0 is outside, v1 is inside
                let t = d0 / (d0 - d1);
                let intersection = v0 + t * (v1 - v0);
                let normal = n0.lerp(n1, t).normalize();
                new_vertices.push(intersection);
                new_normals.push(normal);
            }
        }
        *vertices = new_vertices;
        *normals = new_normals;

        if vertices.len() < 3 {
            return;
        }
    }
}

pub fn entity_has_overlays(entity: &Entity) -> bool {
    entity
        .properties
        .iter()
        .any(|(key, _)| key.starts_with("sides"))
}

pub fn entity_get_overlay_sides<'a>(
    overlay_entity: &'a Entity,
) -> impl Iterator<Item = i32> + 'a {
    overlay_entity
        .properties
        .iter()
        .filter(|(key, _)| key.starts_with("sides"))
        .flat_map(|(_, value)| value.split_whitespace())
        .filter_map(|s| s.parse::<i32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_inside_convex_polygon() {
        let polygon = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        assert!(is_inside_convex_polygon(&Vec3::new(0.5, 0.5, 0.0), &polygon));
        assert!(!is_inside_convex_polygon(&Vec3::new(2.0, 0.5, 0.0), &polygon));
    }

    #[test]
    fn test_clip_polygon_with_normals() {
        let mut vertices = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(-1.0, 2.0, 0.0),
        ];
        let mut normals = vec![
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let clip_polygon = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        clip_polygon_with_normals(&mut vertices, &mut normals, &clip_polygon);

        assert_eq!(vertices.len(), 4);
        assert!(vertices.iter().all(|v| v.x >= 0.0 && v.x <= 1.0 && v.y >= 0.0 && v.y <= 1.0));
    }
}
