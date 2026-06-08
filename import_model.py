from roboflow import Roboflow

rf = Roboflow(api_key="OxMmDugjRblOsY4JZRwb")
project = rf.workspace("charlotte-2ulmc").project("alien-plushie")
version = project.version(1)
dataset = version.download("yolo26")
                