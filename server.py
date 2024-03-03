from bottle import Bottle, request, response, run, static_file,subprocess,os,redirect
import json
from final_test1 import main,Find_Dates
  

app = Bottle()

@app.route('/')
def index():
    # Serve the HTML file
    return static_file('index.html', root='.')

@app.route('/<filename:path>')
def serve_static(filename):
    # Serve static files from the 'static' directory
    return static_file(filename, root='.')
@app.route('/save_username', method='POST')
def save_username():
    username = request.forms.get('username').strip()
    password = request.forms.get('password').strip()
    print(os.path.exists('user_history.txt'))
    with open('user_history.txt', 'a') as file:
        file.write(f"{username},{password}\n")
    return redirect('/')
@app.route('/check_username',method='POST')
def check_username():
    username = request.forms.get('username').strip()
    password=request.forms.get('password').strip()
    with open('user_history.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                saved_username, saved_password = parts
                if username == saved_username:
                    if saved_password == password:
                        pass
    return redirect('/')
@app.post('/process_input')
def process_input():
    data = request.json
    input_value = data.get('ticker_symbol')

    # Process the input value as required
    # For example, you can print it
    print("Received input:", input_value)
    
    # Call the Start function to get computed data
    computed_data = main(input_value)
    dates=Find_Dates(computed_data)
    # Return the computed data as JSON response
    print(computed_data)
    response.content_type = 'application/json'
    return json.dumps(computed_data)
# @app.route('/process_input','POST')
# def calculate():
#     argument = request.form.get('tickerSymbol')
#     process = subprocess.Popen(["python", "final_test.py", argument], stdout=subprocess.PIPE, text=True)
#     output, _ = process.communicate()
#     print("Output from script2.py:", output.strip())

if __name__ == '__main__':
    run(app, host='localhost', port=8080, debug=True)

