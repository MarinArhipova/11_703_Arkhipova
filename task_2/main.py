import requests

if __name__ == '__main__':
    response = requests.get('http://data.kzn.ru:8082/api/v0/dynamic_datasets/bus.json')
    print(response.json()[0]['data'])
    # output: {'GaragNumb': '7137', 'Marsh': '89', 'Graph': '8', 'Smena': '2', 'TimeNav': '12.10.2020 23:15:40',
    # 'Latitude': '55.874383', 'Longitude': '49.002517', 'Speed': '0', 'Azimuth': '174'}

# http://data.kzn.ru/api_doc
