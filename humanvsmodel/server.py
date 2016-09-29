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
    redirect("/question/1")


@route('/thanks')
def thanks():
    return template('thanks.tpl')


@route('/question/<idx>')
def question(idx):
    return template('question', idx=idx)


@route('/save', method='POST')
def save():
    idx = request.forms.get('q_idx')
    wart_x = request.forms.get('wart_x')
    wart_y = request.forms.get('wart_y')
    w_type = request.forms.get('type')
    name = request.cookies.name + ".csv"
    f = open('results/' + name, 'a+')
    # f.seek(0, 2)
    f.write(name + "," + idx + "," + wart_x + "," + wart_y + "," + w_type + "\n")
    f.close()
    if int(idx) == 50:
        redirect("/thanks")
    else:
        redirect("/question/" + str(int(idx) + 1))


@route('/img/<filename>')
def server_static(filename):
    return static_file(filename, root=os.path.join(dir_path, 'images'))

run(host='localhost', port=8080)
