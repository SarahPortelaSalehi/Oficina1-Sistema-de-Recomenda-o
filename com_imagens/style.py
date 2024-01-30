import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("wallpaper.png")

page_style = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
}}

[data-testid="block-container"] {{
background-color: #000000cf;
text-align: center;
}}
"""