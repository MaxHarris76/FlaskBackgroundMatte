from flask import Flask, render_template, request, flash, session, redirect, url_for, send_file
from flask.templating import render_template_string
import os
from numpy.core.fromnumeric import squeeze
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
import random
import string

device = torch.device('cuda')
precision = torch.float16

app = Flask(__name__)
app.secret_key = os.urandom(24)

com_path = ""

@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/login", methods = ["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        user_id = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
        user = user + user_id
        session["user"] = user
        return redirect(url_for("user"))
    else:
        return render_template("login.html")

@app.route("/user")
def user():
    if "user" in session:
        user = session["user"]
        return render_template("submit.html")
    else: 
        return redirect(url_for("login"))



@app.route("/uploader", methods=["GET", "POST"])
def uploader():

    if "user" in session:
        user = session["user"]

        ## Saves and renames images to correct filenames to be processed
        imagefile1 = request.files['imagefile1']
        imagefile1.filename = "bg" + user + ".jpg"
        image1_path = "./input_bg/" + imagefile1.filename
        imagefile1.save(image1_path)

        imagefile2 = request.files['imagefile2']
        imagefile2.filename = "src" + user + ".jpg"
        image2_path = "./input_src/" + imagefile2.filename
        imagefile2.save(image2_path)

        ## Starts matting the source image
        model = torch.jit.load('TorchScript/torchscript_resnet50_fp32.pth')
        model.backbone_scale = 0.25
        model.refine_mode = 'sampling'
        model.refine_sample_pixels = 80_000

        model = model.to(device)

        src_path = 'input_src/src' + user + '.jpg'
        bgr_path = 'input_bg/bg' + user + '.jpg'

        src = Image.open(src_path)
        src = ToTensor()(src).unsqueeze(0)
        src = src.cuda()

        bgr = Image.open(bgr_path)
        bgr = ToTensor()(bgr).unsqueeze(0)
        bgr = bgr.cuda()

        pha, fgr = model(src, bgr)[:2]
        tgt_bgr = torch.tensor([120/255, 255/255, 155/255], device=device).view(1, 3, 1, 1)
        com = fgr * pha + tgt_bgr * (1 - pha)

        unloader = transforms.ToPILImage()

        def image_loader(tensor):
            image = tensor.cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            return image
            
        com = image_loader(com)
        global com_path
        com_path = "static/output" + user + ".jpg"
        com.save(com_path, 'JPEG')

        return render_template("uploader.html")
    else: 
        return redirect(url_for("login"))



@app.route("/view_com", methods=["GET", "POST"])
def view_image():
    if "user" in session:
        user = session["user"]
        return render_template("view_matte.html", img_path = com_path)
    else: 
        return redirect(url_for("login"))

if __name__ =="__main__":
    app.run(debug=True)
