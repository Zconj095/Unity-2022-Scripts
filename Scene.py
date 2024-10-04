import os

class Scene:
    def __init__(self, name):
        self.name = name
        self.objects = []
        self.lights = []
        self.cameras = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def add_camera(self, camera):
        self.cameras.append(camera)
        

class SceneLoader:
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir

    def load_scene(self, scene_name):
        scene_file = os.path.join(self.scene_dir, f"{scene_name}.scene")
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)
        return Scene(scene_data['name'])

    def save_scene(self, scene):
        scene_file = os.path.join(self.scene_dir, f"{scene.name}.scene")
        with open(scene_file, 'w') as f:
            json.dump({'name': scene.name, 'objects': [obj.__dict__ for obj in scene.objects], 'lights': [light.__dict__ for light in scene.lights], 'cameras': [camera.__dict__ for camera in scene.cameras]}, f)

class SceneManager:
    def __init__(self, scene_loader):
        self.scene_loader = scene_loader
        self.current_scene = None

    def switch_scene(self, scene_name):
        self.current_scene = self.scene_loader.load_scene(scene_name)

    def get_current_scene(self):
        return self.current_scene
    
