import tornado.web
import tornado.ioloop
import shutil

from classification_from_dir import predict_img

class uploadImgHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["file"]
        for f in files:
            fh = open(f"predictions/temp/{f.filename}", "wb")
            fh.write(f.body)
            fh.close()
            pred = predict_img(f"predictions/temp/{f.filename}", dir=False)
            shutil.move(f"predictions/temp/{f.filename}", f"predictions/{pred}/{f.filename}")
        self.write(f"http://localhost:1414/predictions/{pred}/{f.filename}")
    def get(self):
        self.render("index.html")

if (__name__ == "__main__"):
    app = tornado.web.Application([
        ("/", uploadImgHandler),
        ("/predictions/temp/(.*)", tornado.web.StaticFileHandler, {'path': 'predictions/temp'})
    ])

    app.listen(1414)
    print("Listening on port 1414")
    tornado.ioloop.IOLoop.instance().start()