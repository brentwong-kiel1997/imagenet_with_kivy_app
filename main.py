from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
import time
import tensorflow as tf

Builder.load_file('frontend.kv')

m = tf.keras.models.load_model('imagenet')

with open('ImageNetLabels.txt', 'r') as file:
    data = file.readlines()


class CameraScreen(Screen):
    def start(self):
        self.ids.camera.play = True
        self.ids.camera_button.text = 'Stop Camera'
        self.ids.camera.opacity = 1

    def stop(self):
        self.ids.camera.play = False
        self.ids.camera_button.text = 'Start Camera'
        self.ids.camera.texture = None
        self.ids.camera.opacity = 0


    def capture(self):
        self.path = 'image/'+ str(time.time_ns())+'image.png'
        self.ids.camera.export_to_png(self.path)
        self.ids.camera.play = False
        self.ids.camera_button.text = 'Start Camera'
        self.ids.camera.texture = None
        self.ids.camera.opacity = 0
        self.manager.current = 'image_screen'
        self.manager.current_screen.ids.img.source = self.path

class ImageScreen(Screen):
    def predict(self):
        img = tf.io.read_file(App.get_running_app().root.ids.camera_screen.path)
        img = tf.io.decode_image(img)
        img = img[60:60 + 720, 160:160 + 1280]
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        p = m.predict(tf.expand_dims(img/255, 0))
        predict = tf.argmax(tf.squeeze(p))
        self.ids.predict_text.text = data[predict].replace('\n', '')
    def back(self):
        self.manager.current = 'camera_screen'
        self.ids.predict_text.text = 'empty'



class RootWidget(ScreenManager):
    pass

class MainApp(App):

    def build(self):
        return RootWidget()


MainApp().run()