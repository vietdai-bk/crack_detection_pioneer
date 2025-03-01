from fastapi import FastAPI, Response
from picamera2 import Picamera2
import uvicorn
from starlette.responses import StreamingResponse
import io
import time

app = FastAPI()

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    lores={"size": (320, 240)},
    encode="main"
)
picam2.configure(camera_config)

def generate_frames():
    picam2.start()
    try:
        while True:
            buffer = io.BytesIO()
            picam2.capture_file(buffer, format='jpeg')
            buffer.seek(0)
            frame = buffer.read()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)
    finally:
        picam2.stop()

@app.get("/video")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/")
async def root():
    html_content = """
    <html>
        <head>
            <title>Raspberry Pi Camera Stream</title>
        </head>
        <body>
            <h1>Raspberry Pi Camera Stream</h1>
            <img src="/video" width="640" height="480">
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)