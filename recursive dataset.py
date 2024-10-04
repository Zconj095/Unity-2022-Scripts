import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = self.normalize(direction)

    def normalize(self, vector):
        return vector / np.linalg.norm(vector)

class Camera:
    def __init__(self, eye, look_at, up, fov, aspect_ratio):
        self.eye = np.array(eye)
        self.look_at = np.array(look_at)
        self.up = np.array(up)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.setup_camera()

    def setup_camera(self):
        self.w = self.normalize(self.eye - self.look_at)
        self.u = self.normalize(np.cross(self.up, self.w))
        self.v = np.cross(self.w, self.u)
        self.half_height = np.tan(np.deg2rad(self.fov) / 2)
        self.half_width = self.aspect_ratio * self.half_height

    def get_ray(self, u, v):
        direction = self.normalize(self.look_at + u * self.half_width * self.u + v * self.half_height * self.v - self.eye)
        return Ray(self.eye, direction)

    def normalize(self, vector):
        return vector / np.linalg.norm(vector)

def refract(ray_direction, normal, eta):
    cosi = -np.dot(normal, ray_direction)
    cost2 = 1 - eta ** 2 * (1 - cosi ** 2)
    if cost2 < 0:
        return None
    refraction_direction = eta * ray_direction + (eta * cosi - np.sqrt(cost2)) * normal
    return refraction_direction

class Sphere:
    def __init__(self, center, radius, refraction_index):
        self.center = np.array(center)
        self.radius = radius
        self.refraction_index = refraction_index

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

def trace_ray(ray, depth, scene):
    if depth > 5:
        return np.array([0, 0, 0])
    
    closest_t = float('inf')
    closest_object = None

    for obj in scene:
        t = obj.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            closest_object = obj

    if closest_object is None:
        return np.array([0.5, 0.7, 1.0])  # Background color

    hit_point = ray.origin + closest_t * ray.direction
    normal = (hit_point - closest_object.center) / np.linalg.norm(hit_point - closest_object.center)
    
    if np.dot(ray.direction, normal) > 0:
        normal = -normal

    refracted_direction = refract(ray.direction, normal, 1 / closest_object.refraction_index)
    if refracted_direction is None:
        return np.array([1, 1, 1])  # Total internal reflection color

    refracted_ray = Ray(hit_point, refracted_direction)
    return trace_ray(refracted_ray, depth + 1, scene)

def render(scene, camera, image_width, image_height):
    aspect_ratio = image_width / image_height
    image = np.zeros((image_height, image_width, 3))
    
    for j in range(image_height):
        for i in range(image_width):
            u = (2 * (i + 0.5) / image_width - 1) * aspect_ratio
            v = (1 - 2 * (j + 0.5) / image_height)
            ray = camera.get_ray(u, v)
            color = trace_ray(ray, 0, scene)
            image[j, i] = np.clip(color, 0, 1)
    
    return image

