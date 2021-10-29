from flask import Flask
from flask_restx import Api, Resource
import numpy as np
import cv2
import glob


app = Flask(__name__)
api = Api(app)


def get_sift_match(query, img):
    
    result = {}
    
    for q in query:
        img1 = cv2.imread(img)
        img2 = cv2.imread(q)
        
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher()
        
        matches = bf.knnMatch(des1,des2, k=2)
        good = []

        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        result[q]=len(good)
        
    result = sorted(result.items(), key=(lambda x:x[1]), reverse=True)
    #print(result)
    return str(next(iter(result))).split('.')[-2].split('_')[1]

def solution():
    qs = glob.glob('query/*.*')
    target_image = 'testImages/16_10.jpeg'
    
    res = get_sift_match(qs, target_image)
   #source = target_image.split('/')[2].split('_')[0]

    return res


@api.route('/imageMatching')
class ImgMatching(Resource):
    def get(self):
        return {"result" : solution()}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)


