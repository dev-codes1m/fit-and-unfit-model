from flask import Flask, request, jsonify

from run import fitUnfitModel

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    bodyImage = request.files['image'].read()

    waistsizebody = int(request.form.get('body-waist-length'))
    chestsizebody = int(request.form.get('body-chest-length'))
    waistsizecloth = int(request.form.get('cloth-waist-length'))
    chestsizecloth = int(request.form.get('cloth-chest-length'))
    result, isFit = fitUnfitModel(bodyImage, chestsizecloth, chestsizebody, waistsizecloth, waistsizebody)
    data = {
        "is_fit": isFit,
        "fitness_score": result
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
