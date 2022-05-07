import requests

url = 'http://localhost:4000/results'
# r = requests.post(url,json={'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400})
r = requests.post(url,json={'sentence':"hello my name is uolive"})

print(r.json())