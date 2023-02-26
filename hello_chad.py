import random
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    image_urls = [
        "https://i.pinimg.com/736x/c6/25/90/c62590c1756680060e7c38011cd704b5.jpg",
        "https://i.pinimg.com/originals/85/9a/f7/859af748d1eed0d67d5801a6df188a89.jpg",
        "https://i.imgflip.com/4ur2wk.png",
        "https://i.imgflip.com/35bdwf.jpg?a465912",
    ]
    random_image_url = random.choice(image_urls)

    return f"""
        <h2 style="color: forestgreen;">Hello Chad</h2>
        <img src="{random_image_url}" alt="Random Image" height=600/>
    """


if __name__ == "__main__":
    app.run()
