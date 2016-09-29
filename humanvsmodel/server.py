from bottle import route, run, template, request, response, static_file, redirect
import os
import time
import random
dir_path = os.path.dirname(os.path.realpath(__file__))


@route('/')
def index():
    return template('intro.tpl')


@route('/start', method='POST')
def start():
    # name = request.forms.get('name')
    name = str(int(time.time()))
    name += "_" + str(random.randint(100, 999))
    response.set_cookie('name', name)
    redirect("/questions/1")


@route('/thanks')
def thanks():
    return template('thanks.tpl')


@route('/questions/<idx>')
def question(idx):
    return template('question', idx=idx)


@route('/save', method='POST')
def save():
    idx = request.forms.get('q_idx')
    wart_x = request.forms.get('wart_x')
    wart_y = request.forms.get('wart_y')
    w_type = request.forms.get('type')
    if w_type is None:
<<<<<<< HEAD
        redirect("/question/" + str(int(idx)))
=======
        redirect("/questions/" + str(int(idx)))
>>>>>>> 1d3a744989cca1c60fa88849170bc2c6d4f5cf99
    else:
        name = request.cookies.name + ".csv"

        f = open('results/' + name, 'a+')
        # f.seek(0, 2)
        f.write(name + "," + idx + "," + wart_x + "," + wart_y + "," + w_type + "\n")
        f.close()
        if int(idx) == 50:
            redirect("/thanks")
        else:
            redirect("/questions/" + str(int(idx) + 1))


@route('/images/<filename>')
def server_static(filename):
    return static_file(filename, root=os.path.join(dir_path, 'images'))


@route('/example/<filename>')
def server_static_example(filename):
    return static_file(filename, root=os.path.join(dir_path, 'examples'))


run(host='0.0.0.0', port=8080)
