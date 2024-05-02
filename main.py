from flask import Flask, render_template, jsonify, redirect, url_for, request
from base64 import b64encode
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
app = Flask(__name__)
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'

def scalarize(x):
    # compute x[...,2]*65536+x[...,1]*256+x[...,0] in efficient way
    y = x[...,2].astype('u4')
    y <<= 8
    y +=x[...,1]
    y <<= 8
    y += x[...,0]
    return y

def topN_colors_v2(img, N):
    img2D = scalarize(img)
    unq,idx,C = np.unique(img2D, return_index=True, return_counts=True)
    topNidx = np.argpartition(C,-N)[-N:]
    return img.reshape(-1,img.shape[-1])[idx[topNidx]], C[topNidx]


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/img', methods=["POST"])
def image():
    imagefile = request.files['img'].stream
    try:
        img = Image.open(imagefile).convert("RGB")
    except:
        return redirect(url_for('home'))
    image = np.array(img)
    # plt.imshow(image)
    image_resh = image.reshape(-1, image.shape[-1])
    color = defaultdict(int)
    for pixel in image_resh:
        rgb = (pixel[0], pixel[1], pixel[2])
        color[rgb] += 1
    sorted_color = sorted(color.items(), key=lambda k_v: k_v[1], reverse=True)[:100]
    colors_hex = []
    for color in sorted_color:
        rgb = color[0]
        colors_hex.append(('%02x%02x%02x' % rgb, color[1]))
    

    i = 0
    while i < len(colors_hex):
        j = i + 1
        while j < len(colors_hex):
            f, s = int(colors_hex[i][0], 16), int(colors_hex[j][0], 16)
            if abs(f - s) < 10000:
                # count = colors_hex[i][1] + colors_hex[j][1]
                # hex_ = hex((f+s)//2)[2:]
                # colors_hex[i] = (colors_hex[i][0], count)
                del colors_hex[j]
            else:
                j += 1
        i += 1
    
    
    sorted_colors_hex = sorted(colors_hex, key=lambda x: x[1], reverse=True)
    sorted_colors_hex = [(sorted_colors_hex[i][0], i+1) for i in range(len(sorted_colors_hex))]
    print(sorted_colors_hex[:10])

    data = io.BytesIO()

    try:
        img.save(data, "JPEG")
    except:
        img.save(data, "PNG")
    
    encoded_img_data = b64encode(data.getvalue())
    decoded_img_data = encoded_img_data.decode('utf-8')
    img_data = f"data:image/jpeg;base64,{decoded_img_data}"
    return render_template("image.html", img = img_data, colors = sorted_colors_hex[:10])

if __name__ == "__main__":
    app.run(debug=True, port=3000)
    