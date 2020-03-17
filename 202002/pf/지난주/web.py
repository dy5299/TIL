from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html', title="my home page")

@app.route('/image')
def image():

    listData = ['123', '456', '789']
    return render_template('image.html')

if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=3000, debug=True)