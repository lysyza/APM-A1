
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import plotly.graph_objects as go
import math
from numpy.linalg import eig
"""
First, we set up the data for our ten bonds
Here, each list corresponds to data from different days, Jan 9th to Jan 20th - ten days of data
Within the list, each bond is characterized by its closing price that day; coupon rate in %; number of coupon payments left;
maturity date
"""
bonds9 = [(96.632, 1.5, 0, '2024-05-01'),(96.1, 1.5, 1,'2024-09-01'),(94.775, 1.25, 2,'2025-03-01'),
          (92.01, 0.5, 3, '2025-09-01'), (90.195, 0.25, 4, '2026-03-01'),(91.704, 1, 5, '2026-09-01'),
          (92.03, 1.25, 6, '2027-03-01'),(91.17, 1, 6, '2027-06-01'),(101.325, 3.5, 8, '2028-03-01'),
          (94.58, 2, 8, '2028-06-01')]
bonds10 = [(96.595, 1.5, 0, '2024-05-01'),(96.105, 1.5, 1,'2024-09-01'),(94.807, 1.25, 2,'2025-03-01'),
           (92.035, 0.5, 3, '2025-09-01'),(90.185, 0.25, 4, '2026-03-01'),(91.705, 1, 5, '2026-09-01'),
           (92.1, 1.25, 6, '2027-03-01'),(91.14, 1, 6, '2027-06-01'),(101.27, 3.5, 8, '2028-03-01'),(94.505, 2, 8, '2028-06-01')]
bonds11 = [(96.671, 1.5, 0, '2024-05-01'), (96.156, 1.5, 1,'2024-09-01'),(94.895, 1.25,2,'2025-03-01'),
           (92.135, 0.5,  3, '2025-09-01'),(90.394, 0.25, 4, '2026-03-01'),(91.93, 1, 5, '2026-09-01'),
           (92.253, 1.25, 6, '2027-03-01'),(91.375, 1, 6, '2027-06-01'),(101.581, 3.5,  8, '2028-03-01'),(94.8, 2,  8, '2028-06-01')]
bonds12 = [
    (96.756, 1.5,0, '2024-05-01'),(96.31, 1.5,  1,'2024-09-01'),(95.071, 1.25, 2,'2025-03-01'),
    (92.355, 0.5, 3, '2025-09-01'),(90.695, 0.25, 4, '2026-03-01'),(92.354, 1, 5, '2026-09-01'),
    (92.69, 1.25, 6, '2027-03-01'),(91.87, 1,  6, '2027-06-01'),(102.228, 3.5,  8, '2028-03-01'),(95.444, 2, 8, '2028-06-01')]
bonds13 = [
    (96.754, 1.5, 0, '2024-05-01'),(96.337, 1.5, 1,'2024-09-01'),(95.18, 1.25, 2,'2025-03-01'),
    (92.5, 0.5, 3, '2025-09-01'),(90.865, 0.25, 4, '2026-03-01'),(92.508, 1,  5, '2026-09-01'),
    (92.955, 1.25, 6, '2027-03-01'),(92.105, 1,6, '2027-06-01'),(102.515, 3.5,  8, '2028-03-01'),(95.63, 2, 8, '2028-06-01')]
bonds16 = [
    (96.8, 1.5,0, '2024-05-01'),(96.44, 1.5, 1,'2024-09-01'),(95.29, 1.25, 2,'2025-03-01'),
    (92.64, 0.5, 3, '2025-09-01'),(90.98, 0.25,  4, '2026-03-01'),(92.623, 1, 5, '2026-09-01'),
    (93.06, 1.25, 6, '2027-03-01'),(92.2, 1, 6, '2027-06-01'),(102.65, 3.5, 8, '2028-03-01'),(95.85, 2, 8, '2028-06-01')]
bonds17 = [
    (96.821, 1.5, 0, '2024-05-01'),(96.439, 1.5, 1,'2024-09-01'),(95.375, 1.25,  2,'2025-03-01'),
    (92.72, 0.5, 3, '2025-09-01'),(91.105, 0.25,  4, '2026-03-01'),(92.749, 1, 5, '2026-09-01'),
    (93.14, 1.25, 6, '2027-03-01'),(92.28, 1, 6, '2027-06-01'),(102.72, 3.5, 8, '2028-03-01'),(95.88, 2, 8, '2028-06-01')]
bonds18 = [
    (96.871, 1.5, 0, '2024-05-01'),(96.61, 1.5,  1,'2024-09-01'),(95.61, 1.25, 2,'2025-03-01'),
    (93.01, 0.5,3, '2025-09-01'),(91.47, 0.25, 4, '2026-03-01'),(93.192, 1,  5, '2026-09-01'),
    (93.69, 1.25, 6, '2027-03-01'),(92.8, 1, 6, '2027-06-01'),(103.355, 3.5,  8, '2028-03-01'),(96.535, 2, 8, '2028-06-01')]
bonds19 = [
    (96.92, 1.5, 0, '2024-05-01'), (96.56, 1.5,  1,'2024-09-01'),(95.65, 1.25,  2,'2025-03-01'),
    (93.12, 0.5, 3, '2025-09-01'), (91.565, 0.25, 4, '2026-03-01'),(93.249, 1, 5, '2026-09-01'),
    (93.73, 1.25,6, '2027-03-01'),(92.795, 1, 6, '2027-06-01'),(103.308, 3.5, 8, '2028-03-01'),(96.518, 2, 8, '2028-06-01')]
bonds20 = [
    (96.87, 1.5, 0, '2024-05-01'),(96.5, 1.5, 1,'2024-09-01'), (95.56, 1.25, 2,'2025-03-01'),
    (93.03, 0.5, 3, '2025-09-01'),(91.45, 0.25, 4, '2026-03-01'),(93.12, 1, 5, '2026-09-01'),
    (93.56, 1.25, 6, '2027-03-01'),(92.6, 1, 6, '2027-06-01'),(103.05, 3.5, 8, '2028-03-01'),(96.27, 2, 8, '2028-06-01')]

"""
QUESTION 1

PART A
"""

"""
We calculate dirty price by adding clean price and accrued interest 
"""
def dirty_price(bond, date):
    dat = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    mat = datetime.datetime.strptime(bond[3], '%Y-%m-%d').date()
    last_coupon = mat - relativedelta(months = (bond[2]+1)*6)
    days_since_last_coupon = (dat - last_coupon).days
    dirty_price = bond[0] + (bond[1]*(1/183)*days_since_last_coupon)
    return dirty_price

"""
Construct a data frame containing cash flows and dates of those cash flows for every bond on every day
"""
def construct_df(bond, date):
    cash_flows = []
    dates = []
    cash_flows.append(-dirty_price(bond, date))
    dates.append(date)
    mat = datetime.datetime.strptime(bond[3], '%Y-%m-%d').date()
    for i in range(0,bond[2]):
        cash_flows.append(bond[1]/2)
        dates.append(mat - relativedelta(months = (bond[2]-i)*6))
    cash_flows.append((bond[1]/2)+100)
    dates.append(bond[3])
    return pd.DataFrame({'cash_flows': cash_flows,'dates': dates})

"""
Calculate YTM
"""
def calculate_xirr(df, initial_guess=0.1):
    df['dates'] = pd.to_datetime(df['dates'])
    dates = df['dates'].tolist()
    start_date = dates[0]

    times = [(d - start_date).days / 365.0 for d in dates]
    cash_flows = df['cash_flows'].tolist()

    rate = initial_guess
    tolerance = 1e-6
    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        npv = sum(cf / (1 + rate) ** t for cf, t in zip(cash_flows, times))

        if abs(npv) < tolerance:
            return rate

        # Adjust the rate based on the sign of the NPV
        rate += npv / 1000 if npv > 0 else npv / 10000
        iteration += 1

    return rate

def ytmlist(bonds, date):
    ytmlist = []
    for bond in bonds:
        df = construct_df(bond, date)
        ytm = calculate_xirr(df)
        ytmlist.append(round(ytm, 4))
    ytmlist1 = np.array(ytmlist)
    return ytmlist1
"""
Here, all YTMs are calculated and added to lists characterized by different days - one for each of the ten days.
"""

ytm_jan9 = ytmlist(bonds9, '2024-01-09')
ytm_jan10 = ytmlist(bonds10, '2024-01-10')
ytm_jan11 = ytmlist(bonds11, '2024-01-11')
ytm_jan12 = ytmlist(bonds12, '2024-01-12')
ytm_jan13 = ytmlist(bonds13, '2024-01-13')
ytm_jan16 = ytmlist(bonds16, '2024-01-16')
ytm_jan17 = ytmlist(bonds17, '2024-01-17')
ytm_jan18 = ytmlist(bonds18, '2024-01-18')
ytm_jan19 = ytmlist(bonds19, '2024-01-19')
ytm_jan20 = ytmlist(bonds20, '2024-01-20')

"""
Now, we plot the graphs superimposed onto each other
"""

maturities = np.array([bond[3] for bond in bonds9])
fig = go.Figure(go.Scatter(x=maturities, y=ytm_jan9 * 100, mode='lines+markers', name ='JAN 9'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan10 * 100, mode='lines+markers', name='JAN 10'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan11 * 100, mode='lines+markers', name='JAN 11'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan12 * 100, mode='lines+markers', name='JAN 12'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan13 * 100, mode='lines+markers', name='JAN 13'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan16 * 100, mode='lines+markers', name='JAN 16'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan17 * 100, mode='lines+markers', name='JAN 17'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan18 * 100, mode='lines+markers', name='JAN 18'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan19 * 100, mode='lines+markers', name='JAN 19'))
fig.add_trace(go.Scatter(x=maturities, y=ytm_jan20 * 100, mode='lines+markers', name='JAN 20'))
fig.update_layout(title='Yield Curve', xaxis_title='Maturity Dates', yaxis_title='YTMs (%)')
#fig.show()


"""
QUESTION 1

PART B
"""

"""
Calculate Spot Rates of every bond for every day
"""
def spot_rate_list1(bonds, date):
    dat = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    spot_rate_list_with_n = []
    for bond in bonds:
        mat = datetime.datetime.strptime(bond[3], '%Y-%m-%d').date()
        diffyears = mat.year - dat.year
        difference = mat - dat.replace(mat.year)
        n = diffyears + (difference.days + difference.seconds / 86400.0) / 365
        if bond[2] == 0:
            a = math.log(dirty_price(bond, date) / (100 + (bond[1] / 2)))
            rate = -a / n
            tup = (rate,n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 1:
            pwr = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 2:
            pwr1 = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            pwr2 = -spot_rate_list_with_n[1][0] * spot_rate_list_with_n[1][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr1)) - ((bond[1] / 2) * math.exp(pwr2))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 3:
            pwr1 = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            pwr2 = -spot_rate_list_with_n[1][0] * spot_rate_list_with_n[1][1]
            pwr3 = -spot_rate_list_with_n[2][0] * spot_rate_list_with_n[2][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr1)) - ((bond[1] / 2) * math.exp(pwr2)) - ((bond[1] / 2) * math.exp(pwr3))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 4:
            pwr1 = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            pwr2 = -spot_rate_list_with_n[1][0] * spot_rate_list_with_n[1][1]
            pwr3 = -spot_rate_list_with_n[2][0] * spot_rate_list_with_n[2][1]
            pwr4 = -spot_rate_list_with_n[3][0] * spot_rate_list_with_n[3][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr1)) - ((bond[1] / 2) * math.exp(pwr2)) - (
                  (bond[1] / 2) * math.exp(pwr3)) - ((bond[1] / 2) * math.exp(pwr4))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 5:
            pwr1 = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            pwr2 = -spot_rate_list_with_n[1][0] * spot_rate_list_with_n[1][1]
            pwr3 = -spot_rate_list_with_n[2][0] * spot_rate_list_with_n[2][1]
            pwr4 = -spot_rate_list_with_n[3][0] * spot_rate_list_with_n[3][1]
            pwr5 = -spot_rate_list_with_n[4][0] * spot_rate_list_with_n[4][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr1)) - ((bond[1] / 2) * math.exp(pwr2)) - (
              (bond[1] / 2) * math.exp(pwr3)) - ((bond[1] / 2) * math.exp(pwr4)) - ((bond[1] / 2) * math.exp(pwr5))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 6:
            pwr1 = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            pwr2 = -spot_rate_list_with_n[1][0] * spot_rate_list_with_n[1][1]
            pwr3 = -spot_rate_list_with_n[2][0] * spot_rate_list_with_n[2][1]
            pwr4 = -spot_rate_list_with_n[3][0] * spot_rate_list_with_n[3][1]
            pwr5 = -spot_rate_list_with_n[4][0] * spot_rate_list_with_n[4][1]
            pwr6 = -spot_rate_list_with_n[5][0] * spot_rate_list_with_n[5][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr1)) - ((bond[1] / 2) * math.exp(pwr2)) - (
                    (bond[1] / 2) * math.exp(pwr3)) - ((bond[1] / 2) * math.exp(pwr4)) - (
                            (bond[1] / 2) * math.exp(pwr5)) - ((bond[1] / 2) * math.exp(pwr6))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
        elif bond[2] == 8:
            pwr1 = -spot_rate_list_with_n[0][0] * spot_rate_list_with_n[0][1]
            pwr2 = -spot_rate_list_with_n[1][0] * spot_rate_list_with_n[1][1]
            pwr3 = -spot_rate_list_with_n[2][0] * spot_rate_list_with_n[2][1]
            pwr4 = -spot_rate_list_with_n[3][0] * spot_rate_list_with_n[3][1]
            pwr5 = -spot_rate_list_with_n[4][0] * spot_rate_list_with_n[4][1]
            pwr6 = -spot_rate_list_with_n[5][0] * spot_rate_list_with_n[5][1]
            pwr7 = -spot_rate_list_with_n[6][0] * spot_rate_list_with_n[6][1]
            pwr8 = -spot_rate_list_with_n[7][0] * spot_rate_list_with_n[7][1]
            x = dirty_price(bond, date) - ((bond[1] / 2) * math.exp(pwr1)) - ((bond[1] / 2) * math.exp(pwr2)) - (
                    (bond[1] / 2) * math.exp(pwr3)) - ((bond[1] / 2) * math.exp(pwr4)) - (
                        (bond[1] / 2) * math.exp(pwr5)) - ((bond[1] / 2) * math.exp(pwr6)) - ((bond[1] / 2) * math.exp(pwr7)) -((bond[1] / 2) * math.exp(pwr8))
            y = 1 / (100 + (bond[1] / 2))
            rate = -math.log(x * y) / n
            tup = (rate, n)
            spot_rate_list_with_n.append(tup)
    return spot_rate_list_with_n

def make_spot_rate_list(bonds, date):
    spot_rate_list = []
    biglist = spot_rate_list1(bonds, date)
    for i in range(10):
        spot_rate_list.append(biglist[i][0])
    return spot_rate_list

"""
Here, spot rates are all calculated and sorted into lists by date.
"""
spot9 = make_spot_rate_list(bonds9, '2024-01-09')
spot10 = make_spot_rate_list(bonds10, '2024-01-10')
spot11 = make_spot_rate_list(bonds11, '2024-01-11')
spot12 = make_spot_rate_list(bonds12, '2024-01-12')
spot13 = make_spot_rate_list(bonds13, '2024-01-13')
spot16 = make_spot_rate_list(bonds16, '2024-01-16')
spot17 = make_spot_rate_list(bonds17, '2024-01-17')
spot18 = make_spot_rate_list(bonds18, '2024-01-18')
spot19 = make_spot_rate_list(bonds19, '2024-01-19')
spot20 = make_spot_rate_list(bonds20, '2024-01-20')

"""
Plot the graphs same as before.
"""
fig = go.Figure(go.Scatter(x=maturities, y=spot9 * 100, mode='lines+markers', name ='JAN 9'))
fig.add_trace(go.Scatter(x=maturities, y=spot10 * 100, mode='lines+markers', name='JAN 10'))
fig.add_trace(go.Scatter(x=maturities, y=spot11 * 100, mode='lines+markers', name='JAN 11'))
fig.add_trace(go.Scatter(x=maturities, y=spot12 * 100, mode='lines+markers', name='JAN 12'))
fig.add_trace(go.Scatter(x=maturities, y=spot13 * 100, mode='lines+markers', name='JAN 13'))
fig.add_trace(go.Scatter(x=maturities, y=spot16 * 100, mode='lines+markers', name='JAN 16'))
fig.add_trace(go.Scatter(x=maturities, y=spot17 * 100, mode='lines+markers', name='JAN 17'))
fig.add_trace(go.Scatter(x=maturities, y=spot18 * 100, mode='lines+markers', name='JAN 18'))
fig.add_trace(go.Scatter(x=maturities, y=spot19 * 100, mode='lines+markers', name='JAN 19'))
fig.add_trace(go.Scatter(x=maturities, y=spot20 * 100, mode='lines+markers', name='JAN 20'))
fig.update_layout(title='Spot Rate Curve', xaxis_title='Maturity Dates', yaxis_title='Spot Rates (%)')
#fig.show()

"""
QUESTION 1

PART C
"""

"""
Use the formula from hints to calculate forward rates.
"""
def calc_forward_rate(spot_rate1, spot_rate2, years_ro_mat):
    x = (1+spot_rate2)**(2*years_ro_mat+2)
    y = (1+spot_rate1)**(2*years_ro_mat)
    f = ((x/y)**0.5) -1
    return f

def get_forward_rates(bonds, date):
    spot_rates = spot_rate_list1(bonds, date)
    f1 = calc_forward_rate(spot_rates[2][0], spot_rates[4][0], spot_rates[2][1])
    f2 = calc_forward_rate(spot_rates[3][0], spot_rates[5][0], spot_rates[3][1])
    f3 = calc_forward_rate(spot_rates[4][0], spot_rates[6][0], spot_rates[4][1])
    f4 = calc_forward_rate(spot_rates[6][0], spot_rates[8][0], spot_rates[6][1])
    f5 = calc_forward_rate(spot_rates[7][0], spot_rates[9][0], spot_rates[7][1])
    forward_rates = [f1, f2, f3, f4, f5]
    return forward_rates

"""
Combine the forward rates into lists sorted by date and plot the curves same as before.
"""
frw9 = get_forward_rates(bonds9, '2024-01-09')
frw10 = get_forward_rates(bonds10, '2024-01-10')
frw11 = get_forward_rates(bonds11, '2024-01-11')
frw12 = get_forward_rates(bonds12, '2024-01-12')
frw13 = get_forward_rates(bonds13, '2024-01-13')
frw16 = get_forward_rates(bonds16, '2024-01-16')
frw17 = get_forward_rates(bonds17, '2024-01-17')
frw18 = get_forward_rates(bonds18, '2024-01-18')
frw19 = get_forward_rates(bonds19, '2024-01-19')
frw20 = get_forward_rates(bonds20, '2024-01-20')

maturities2 = maturities[2:]

fig = go.Figure(go.Scatter(x=maturities2, y=frw9 * 100, mode='lines+markers', name ='JAN 9'))
fig.add_trace(go.Scatter(x=maturities2, y=frw10 * 100, mode='lines+markers', name='JAN 10'))
fig.add_trace(go.Scatter(x=maturities2, y=frw11 * 100, mode='lines+markers', name='JAN 11'))
fig.add_trace(go.Scatter(x=maturities2, y=frw12 * 100, mode='lines+markers', name='JAN 12'))
fig.add_trace(go.Scatter(x=maturities2, y=frw13 * 100, mode='lines+markers', name='JAN 13'))
fig.add_trace(go.Scatter(x=maturities2, y=frw16 * 100, mode='lines+markers', name='JAN 16'))
fig.add_trace(go.Scatter(x=maturities2, y=frw17 * 100, mode='lines+markers', name='JAN 17'))
fig.add_trace(go.Scatter(x=maturities2, y=frw18 * 100, mode='lines+markers', name='JAN 18'))
fig.add_trace(go.Scatter(x=maturities2, y=frw19 * 100, mode='lines+markers', name='JAN 19'))
fig.add_trace(go.Scatter(x=maturities2, y=frw20 * 100, mode='lines+markers', name='JAN 20'))
fig.update_layout(title='Forward Rate Curve', xaxis_title='Maturity Dates', yaxis_title='Forward Rates (%)')
#fig.show()

"""
QUESTION 2
"""

"""
These are lists of all YTMs and forward dates, sorted by date.
"""
ytms = [ytm_jan9, ytm_jan10, ytm_jan11, ytm_jan12, ytm_jan13, ytm_jan16, ytm_jan17, ytm_jan18,ytm_jan19, ytm_jan20]
frws = [frw9, frw10, frw11, frw12, frw13, frw16, frw17, frw18, frw19, frw20]

"""
Now we use the formula provided in the assignment to calculate log-returns.
"""
def log_returns1(bondnumb, date):
    days = [9, 10,11,12,13,16,17,18,19,20]
    day = int(date[-2:])
    ind = days.index(day)
    r1 = ytms[ind][bondnumb]
    r2 = ytms[ind+1][bondnumb]
    div = r2/r1
    return math.log(div)

"""
Building a log-return matrix for every other bond - every bond that matures in spring in all five years.
"""
log_return_matrix = np.array([
    [log_returns1(0, '2024-01-09'), log_returns1(0, '2024-01-10'), log_returns1(0, '2024-01-11'),
     log_returns1(0, '2024-01-12'), log_returns1(0, '2024-01-13'), log_returns1(0, '2024-01-16'),
     log_returns1(0, '2024-01-17'), log_returns1(0, '2024-01-18'), log_returns1(0, '2024-01-19')],
    [log_returns1(2, '2024-01-09'), log_returns1(2, '2024-01-10'), log_returns1(2, '2024-01-11'),
     log_returns1(2, '2024-01-12'), log_returns1(2, '2024-01-13'), log_returns1(2, '2024-01-16'),
     log_returns1(2, '2024-01-17'), log_returns1(2, '2024-01-18'), log_returns1(2, '2024-01-19')],
    [log_returns1(4, '2024-01-09'),log_returns1(4, '2024-01-10'), log_returns1(4, '2024-01-11'),
     log_returns1(4, '2024-01-12'), log_returns1(4, '2024-01-13'), log_returns1(4, '2024-01-16'),
     log_returns1(4, '2024-01-17'), log_returns1(4, '2024-01-18'), log_returns1(4, '2024-01-19')],
    [log_returns1(6, '2024-01-09'),log_returns1(6, '2024-01-10'), log_returns1(6, '2024-01-11'),
     log_returns1(6, '2024-01-12'), log_returns1(6, '2024-01-13'), log_returns1(6, '2024-01-16'),
     log_returns1(6, '2024-01-17'), log_returns1(6, '2024-01-18'), log_returns1(6, '2024-01-19')],
    [log_returns1(8, '2024-01-09'),log_returns1(8, '2024-01-10'), log_returns1(8, '2024-01-11'),
     log_returns1(8, '2024-01-12'), log_returns1(8, '2024-01-13'), log_returns1(8, '2024-01-16'),
     log_returns1(8, '2024-01-17'), log_returns1(8, '2024-01-18'), log_returns1(8, '2024-01-19')]
])

"""
Construct a covariance matrix.
"""
cov_matrix1 = np.cov(log_return_matrix)
#print(cov_matrix1)

"""
We repeat the same process for the forward rates.
"""
def log_returns2(numb, date):
    days = [9, 10,11,12,13,16,17,18,19,20]
    day = int(date[-2:])
    ind = days.index(day)
    r1 = frws[ind][numb]
    r2 = frws[ind+1][numb]
    div = r2/r1
    return math.log(div)

log_return_matrix2 = np.array([
    [log_returns2(0, '2024-01-09'), log_returns2(0, '2024-01-10'), log_returns2(0, '2024-01-11'),
     log_returns2(0, '2024-01-12'), log_returns2(0, '2024-01-13'), log_returns2(0, '2024-01-16'),
     log_returns2(0, '2024-01-17'), log_returns2(0, '2024-01-18'), log_returns2(0, '2024-01-19')],
    [log_returns2(1, '2024-01-09'), log_returns2(1, '2024-01-10'), log_returns2(1, '2024-01-11'),
     log_returns2(1, '2024-01-12'), log_returns2(1, '2024-01-13'), log_returns2(1, '2024-01-16'),
     log_returns2(1, '2024-01-17'), log_returns2(1, '2024-01-18'), log_returns2(1, '2024-01-19')],
    [log_returns2(2, '2024-01-09'),log_returns2(2, '2024-01-10'), log_returns2(2, '2024-01-11'),
     log_returns2(2, '2024-01-12'), log_returns2(2, '2024-01-13'), log_returns2(2, '2024-01-16'),
     log_returns2(2, '2024-01-17'), log_returns2(2, '2024-01-18'), log_returns2(2, '2024-01-19')],
    [log_returns2(3, '2024-01-09'),log_returns2(3, '2024-01-10'), log_returns2(3, '2024-01-11'),
     log_returns2(3, '2024-01-12'), log_returns2(3, '2024-01-13'), log_returns2(3, '2024-01-16'),
     log_returns2(3, '2024-01-17'), log_returns2(3, '2024-01-18'), log_returns2(3, '2024-01-19')],
    [log_returns2(4, '2024-01-09'),log_returns2(4, '2024-01-10'), log_returns2(4, '2024-01-11'),
     log_returns2(4, '2024-01-12'), log_returns2(4, '2024-01-13'), log_returns2(4, '2024-01-16'),
     log_returns2(4, '2024-01-17'), log_returns2(4, '2024-01-18'), log_returns2(4, '2024-01-19')]
])

cov_matrix2 = np.cov(log_return_matrix2)
#print(cov_matrix2)

"""
QUESTION 3
"""

"""
Here we calculate the eigenvalues and eigenvectors for both matrices.
"""
w,v=eig(cov_matrix1)
#print('Eigenvalues:', w)
#print('Eigenvectors', v)

a,b=eig(cov_matrix2)
#print('Eigenvalues:', a)
#print('Eigenvectors', b)




