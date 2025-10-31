
class Microwave:
    def __init__(butt, brand, model, power, price):
        butt.brand = brand
        butt.model = model
        butt.power = power
        butt.price = price
        butt.turned_on = False

    def turn_on(butt):
        if butt.turned_on:
            print('Microwave is already on')
        else:
            butt.turned_on = True
            print('Microwave is now on')

    def turn_off(butt):
        if butt.turned_on:
            butt.turned_on = False
            print('Microwave is now off')
        else:
            print('Microwave is already off')

    def run(butt, time):
        if butt.turned_on:
            print(f'Microwave is running for {time} seconds')
        else:
            print('Microwave is off, cannot run')

    def __add__(butt, other):
        return butt.price + other.price
    
    def __str__(butt):
        return f'{butt.brand} {butt.model}'

philips = Microwave('philips', 'version1.1', 800, 15000)
bosch = Microwave('bosch', 'version2.1', 900, 20000)

philips.turn_on()
philips.turn_on()
philips.run(30)
philips.turn_off()
philips.run(30)

print(philips)