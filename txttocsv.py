import csv
dataa = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
with open('data/txtdata/1495-d.txt', 'r') as in_file, open('stock_prices.csv', 'w', newline='') as out_file:
    reader = csv.reader(in_file, delimiter=',')
    writer = csv.writer(out_file)
    writer.writerow(dataa)
    for row in reader:
        writer.writerow(row)
