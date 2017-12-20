# -*- coding: utf-8 -*-

from collections import namedtuple
import re
import os
import calendar

import itertools

import pandas as pd, numpy as np
from scipy import  stats

from numpy.random import randint
from IPython.display import HTML, Javascript

from sklearn.linear_model import LinearRegression
from numpy import *

from copy import deepcopy

# ========================================================================

# директория с данными
data_dir = './data/'

# паттерны для имен файлов
data_file_patt = 'yellow_tripdata_(?P<year>.*?)-(?P<month>.*?).csv'
data_file_name = 'yellow_tripdata_{year}-{month:02}.csv'

agg_file_patt = '(?P<aggregate_by>.*?)_?agg_(?P<year>.*?)-(?P<month>.*?).csv'
agg_file_name = '{aggregate_by}agg_{year}-{month:02}.csv'

# ========================================================================

YOUR_API_KEY='AIzaSyANmGkaFLuUPwnD6qwpSZVCSTnL533C1hc'

# ========================================================================

# Количество регрессоров для модели AGGLAR
NHOURS = 6

# ========================================================================

def _g(c,i):
    return c[i] if isinstance(c,(list,tuple)) else c

class LatLong(namedtuple('LatLong', 'lat long')):

    def __add__(self, c):
        return LatLong(*[self[i]+_g(c,i) for i in [0,1]])

    def __sub__(self, c):
        return LatLong(*[self[i]-_g(c,i) for i in [0,1]])
    
    def __mul__(self, c):
        return LatLong(*[1.0*self[i]*_g(c,i) for i in [0,1]])
    
    def __truediv__(self, c):
        return LatLong(*[1.0*self[i]/_g(c,i) for i in [0,1]])

# ========================================================================

# количество интервалов разбиения по одной оси
N = 50

# координаты региона с которым придется работать
NY = [LatLong(40.49612, -74.25559), LatLong(40.91553, -73.70001)]

# размер одной ячейки
RS = (NY[1]-NY[0])/N

# координаты Empire Street Building
ESB = LatLong(40.748404, -73.985721)

# id зоны (одна из отобранных в предыдущем задании) с которой будем работать в этом задании
# собственно зона с Empire Street Building
RID = 1231 

# ========================================================================
### Регионы (зоны) - нумерация и другие функции

# returns region id by lat lng
def regid_by_ll(ll):
    ll = LatLong(*ll)
    nll = (ll - NY[0])/(NY[1] - NY[0]) * N
    a = lambda x: int(x) if int(x)<N else N-1
    return a(nll.lat) + a(nll.long) * N + 1    

# returns region id by number on lat/lng, where lat/lng is from 1 to N
def regid_by_nll(nll):
    return nll[0] + (nll[1]-1)*N

# returns region rectangle by number on lat/lng, where lat/lng is from 1 to N
def reg_by_nll(nll):
    return [NY[0] + RS * (nll[0]-1, nll[1]-1), NY[0] + RS * (nll[0], nll[1])]

def nll_by_regid(regid):
    return (regid-1)%N + 1, (regid-1)/N + 1

# ========================================================================
### Загрузка и обработка данных

def aggregate_data(year, month, aggreagate_by=''):

    filename = data_dir + data_file_name.format(year = year, month = month)

    ndays  = calendar.monthrange(year, month)[1]
    nhours = ndays*24
    date0 = np.datetime64('{year}-{month:02}-01T00:00:00'.format(year = year, month = month))

    # Загружаем файл
    df=pd.read_csv(filename, ',', parse_dates=[1,2])

    # Уберем поездки нулевой длительности
    df = df[(df.tpep_dropoff_datetime - df.tpep_pickup_datetime)>np.timedelta64(0,'s')]

    # уберем поездки без пассажиров
    df = df[df.passenger_count > 0]

    # уберем поездки c нулевым расстоянием по счетчику
    df = df[df.trip_distance > 0.0]

    #отфильтруем так же сразу поездки по координатам
    df = df[
        (df.pickup_latitude>=NY[0].lat) & 
        (df.pickup_latitude<=NY[1].lat) & 
        (df.pickup_longitude>=NY[0].long) & 
        (df.pickup_longitude<=NY[1].long)
    ]


    # Рассортируем по зонам, посчитав общее кооличество поездок из каждой ячейки
    r = stats.binned_statistic_2d(df.pickup_latitude, df.pickup_longitude,
                                  None,'count',(N, N),
                                  [[NY[0].lat, NY[1].lat], [NY[0].long, NY[1].long]],
                                  expand_binnumbers=True)

    # Запишем номер зоны в поле region_id
    df['region_id'] = map(regid_by_nll, r.binnumber.T)


    # Округляем время посадки до часов и записываем в новую колонку
    df['hour'] = df.tpep_pickup_datetime.dt.floor('H')

    # Построим пустую таблицу с часами по строкам и номером региона по столбцам. 
    agg=pd.DataFrame(
        index=[date0+np.timedelta64(1,'h')*i for i in range(nhours)],
        columns = range(1, N*N+1)
    ).fillna(0.0)

    # Группируем исходную таблицу по часам и номеру региона
    # Подсчитываем количество объектов в каждой группе
    # Разворачиваем (unstack) таблицу, превращая иерархический индекс в индексы столбов и строк
    gb = df.groupby(['hour','region_id'])
    agg2 = gb.size() if aggreagate_by=='' else gb[aggreagate_by].mean()
    agg2 = agg2.unstack()

    # Записываем полученные данные в таблицу, которую мы создали выше
    agg.update(agg2)
    
    return agg

def _sep(aggregate_by): return aggregate_by + '_' if aggregate_by else aggregate_by

def process_data(aggregate_by = ''):

    # найдем все файлы с данными в рабочей дериктории и превратим в список пар [год,месяц]
    dates = [m.groupdict() for m in [re.match(data_file_patt, f) for f in os.listdir(data_dir)] if m]
    dates = [map(int, [d['year'], d['month']]) for d in dates]
    
    for year, month in dates:

        dfn = data_file_name.format(year=year, month=month)
        afn = agg_file_name.format(year=year, month=month, aggregate_by = _sep(aggregate_by))
        
        if not os.path.exists(data_dir + afn):
            
            print ('Processing ', dfn, 'for', afn, '...',)
            df = aggregate_data(year, month, aggregate_by)
            df.to_csv(data_dir + afn)
            print ('Done')
        
        else:

            print (afn, 'is up to date')

# load aggreagated data for year, month and (n-1) months back to past
# if n is 0 load all past data
def load_aggregated(n=-1, year=2016, month=5, aggregate_by=None, verbose = True):
    
    aggregate_by = aggregate_by if aggregate_by is not None else ''
    
    n = n if n>=0 else 1000000 # заведомо большое число файлов
    
    df = None
    for i in range(n):
        afn = agg_file_name.format(year=year, month=month, aggregate_by = _sep(aggregate_by))
        if not os.path.exists(data_dir + afn): break
        if verbose: print ('Loading ', afn, '...', )
        df2 = pd.read_csv(data_dir + afn, index_col=0, parse_dates=True)
        df2.columns = map(int, df2.columns)
        df = df2 if df is None else pd.concat([df2,df])
        if verbose: print ('Done')
        
        month -=1
        if month == 0:
            month = 12
            year -= 1

    return df

# ========================================================================
def basic_template(template, tdict = {}):
    
    def sub(match):
        obj = match.group(1)
        ind = None
        attr = None

        m = re.match('(.*)\[(.*)\]', obj)
        if m:
            obj = m.group(1)
            ind = int(m.group(2))
        else:
            m = re.match('(.*)\.(.*)', obj)
            if m:
                obj = m.group(1)
                attr = m.group(2)

        obj = tdict[obj] if obj in tdict else globals()[obj]
        
        # print "O:", obj, ind, attr

        obj = obj[ind] if ind is not None else getattr(obj, attr) if attr is not None else obj
    
        return str(obj)
    
    return re.sub('{{(.*?)}}', sub, template)


# ========================================================================

def google_map(center = ESB, zoom = 10, size = (400, 400), title = '',
               markers = [], rectangles = [],
               box = NY, box_color = '#FF0000', box_action = '', box_marker = ESB,
               map_action = None,
               verbose = False):
   
    # map_action is called when bounds of map are changed
    # map_action(zoom, center_lat, center_lng)
    
    map_action_html = '''
        map.addListener('bounds_changed', function() {
            c = map.getCenter();
            z = map.getZoom();
            var cmd = '%s(' + z + ',' + c.lat() + ',' + c.lng() + ')';
            IPython.notebook.kernel.execute(cmd);
        });
    
    '''%map_action if map_action else ''
    
    box_template = '''
        var box = new google.maps.Rectangle({
            strokeWeight: 2,
            strokeColor: '{{box_color}}',
            strokeOpacity: 1,
            fillOpacity: 0,
            map: map,
            bounds: {
                south: {{south}},
                west:  {{west}},
                north: {{north}},
                east:  {{east}}
            }
        });
        
        var marker = new google.maps.Marker({
              position: {lat: {{box_marker[0]}}, lng: {{box_marker[1]}} },
              map: map,
        });
        
        box.addListener('dblclick', function(e) {
            marker.setPosition(e.latLng);
            var cmd = '{{box_action}}('+e.latLng.lat()+','+e.latLng.lng()+')';
            IPython.notebook.kernel.execute(cmd);
        });
    '''
    box_html = ''
    if box:
        (south, west), (north, east) = box
        box_html = basic_template(box_template, locals())
    
    marker_template = '''
        new google.maps.Marker({
              position: {lat: {{marker[0]}}, lng: {{marker[1]}} },
              map: map,
        });
    '''    
    markers_html = ''.join([basic_template(marker_template, locals()) for marker in markers])
        
    rect_template = '''
        new google.maps.Rectangle({
            strokeWeight: 0,
            fillColor: '{{color}}',
            fillOpacity: {{opacity}},
            clickable: false,
            map: map,
            bounds: {
                south: {{south}},
                north: {{north}},
                east:  {{east}},
                west:  {{west}}
            }
        });
    '''    
    rect_html = ''.join([
        basic_template(rect_template, locals()) 
        for ((south, west), (north, east)), color, opacity in rectangles
    ])

    
    
    
    template = '''
<!DOCTYPE html>
<html>
  <head>
    <style>
       #map{{map}} {
        height: {{size[1]}}px;
        width: {{size[0]}}px;
       }
    </style>
  </head>
  <body>
    {{title}}
    <div id="map{{map}}"></div>
    <script>
      function initMap() {
        var map = new google.maps.Map(document.getElementById('map{{map}}'), {
          zoom: {{zoom}},
          disableDoubleClickZoom: true, 
          center: {lat: {{center[0]}}, lng: {{center[1]}} }
        });
        
        {{map_action_html}}
        
        {{markers_html}}
        {{rect_html}}
        {{box_html}}
      }
    </script>
    
    
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={{YOUR_API_KEY}}&callback=initMap">
    </script>
    
  </body>
</html>
    '''
    if title: title = basic_template('<h4>{{title}}</h4>', locals())
    map = randint(100000)
    
    html = basic_template(template, locals())
    
    if verbose: print (html)

    return HTML(html)


# ========================================================================

def beep():
    return Javascript('''
        function beep() {
            var snd = new Audio("data:audio/wav;base64,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qJrx4KOQjahZxWbcZgztj2c49nKmkId44S71j0c8eV9yDK6uPRzx5X18eDvjvQ6yKo9ZSS6l//8elePK/Lf//IInrOF/FvDoADYAGBMGb7FtErm5MXMlmPAJQVgWta7Zx2go+8xJ0UiCb8LHHdftWyLJE0QIAIsI+UbXu67dZMjmgDGCGl1H+vpF4NSDckSIkk7Vd+sxEhBQMRU8j/12UIRhzSaUdQ+rQU5kGeFxm+hb1oh6pWWmv3uvmReDl0UnvtapVaIzo1jZbf/pD6ElLqSX+rUmOQNpJFa/r+sa4e/pBlAABoAAAAA3CUgShLdGIxsY7AUABPRrgCABdDuQ5GC7DqPQCgbbJUAoRSUj+NIEig0YfyWUho1VBBBA//uQZB4ABZx5zfMakeAAAAmwAAAAF5F3P0w9GtAAACfAAAAAwLhMDmAYWMgVEG1U0FIGCBgXBXAtfMH10000EEEEEECUBYln03TTTdNBDZopopYvrTTdNa325mImNg3TTPV9q3pmY0xoO6bv3r00y+IDGid/9aaaZTGMuj9mpu9Mpio1dXrr5HERTZSmqU36A3CumzN/9Robv/Xx4v9ijkSRSNLQhAWumap82WRSBUqXStV/YcS+XVLnSS+WLDroqArFkMEsAS+eWmrUzrO0oEmE40RlMZ5+ODIkAyKAGUwZ3mVKmcamcJnMW26MRPgUw6j+LkhyHGVGYjSUUKNpuJUQoOIAyDvEyG8S5yfK6dhZc0Tx1KI/gviKL6qvvFs1+bWtaz58uUNnryq6kt5RzOCkPWlVqVX2a/EEBUdU1KrXLf40GoiiFXK///qpoiDXrOgqDR38JB0bw7SoL+ZB9o1RCkQjQ2CBYZKd/+VJxZRRZlqSkKiws0WFxUyCwsKiMy7hUVFhIaCrNQsKkTIsLivwKKigsj8XYlwt/WKi2N4d//uQRCSAAjURNIHpMZBGYiaQPSYyAAABLAAAAAAAACWAAAAApUF/Mg+0aohSIRobBAsMlO//Kk4soosy1JSFRYWaLC4qZBYWFRGZdwqKiwkNBVmoWFSJkWFxX4FFRQWR+LsS4W/rFRb/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////VEFHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU291bmRib3kuZGUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAwNGh0dHA6Ly93d3cuc291bmRib3kuZGUAAAAAAAAAACU=");snd.play();}
            beep();
    ''')


# ========================================================================
### Признаки - периодические, погодные и т.п.
#### Периодические

def gen_sincos_f(ts, *args):
    '''
    args: list of tuples (k, period)
    '''
    
    args = [(k,p) for (k,p) in args if k>0 and p>0]

    series = [
        pd.Series(
            data = (sin if sine else cos)( arange(len(ts)) * 2*pi * i / p ),
            name = (('S' if sine else 'C') +'_p{}k{}i{}').format(p, k, i),
            index = ts.index
        )
        for k, p in args
        for i in range(1, k+1) 
        for sine in [True, False] 
    ]
    
    return pd.concat(series, axis=1) if series else pd.DataFrame(index=ts.index)

# ========================================================================
#### Признаки выходного дня
'''
# require holiday module, so commented to enforce notebook to run with default libs

def gen_holidays_f(ts):
    index = pd.Series(index = ts.index).resample('D').count().index
    h = holidays.US()
    return pd.DataFrame(
        [float((d in h) or d.weekday() in [5,6]) for d in index], 
        index = index,
        columns = ['Holiday']
    ).reindex(ts.index, method = 'pad')
'''

def gen_holidays_f(ts):
    index = pd.Series(index = ts.index).resample('D').count().index
    return pd.DataFrame(
        [float(d.weekday() in [5,6]) for d in index], 
        index = index,
        columns = ['Holiday']
    ).reindex(ts.index, method = 'pad')

# ========================================================================
#### Погода
# В таблице 'NY_weather.csv' предварительно агрегированная история погоды с сайта wunderweather.

def gen_weather_f(ts):
    
    df = pd.read_csv('NY_weather.csv', index_col=0, parse_dates=True, keep_default_na=False)
    
    t_features = ['Temp_AA', 'Precipitation', 'Pressure', 'Temp', 'Humidity', 'Wind', 'Visibility']
    
    df = df[t_features]
    
    df.Temp_AA = abs(df.Temp_AA)
    
    for f in ['Pressure', 'Temp', 'Humidity', 'Wind', 'Visibility']: 
        df[f] = (df[f] - df[f].mean()) / df[f].std()

    for f in ['Temp_AA', 'Precipitation']: 
        df[f] = df[f] / df[f].std()
        
    return df.reindex(ts.index, method = 'pad')


# ========================================================================
#### Шорткат для получения всех признаков в одной таблице

def gen_exog(ts, yk=0, wk=0, dk=0, weather = True, holidays = True, wh_hd=None):
    if wh_hd is not None:
        weather = wh_hd
        holidays = wh_hd
    w = gen_weather_f(ts) if weather else None
    h = gen_holidays_f(ts) if holidays else None
    return pd.concat([gen_sincos_f(ts, [yk, 24*365], [wk, 27*7], [dk, 24]), w, h], axis = 1)

# ========================================================================
# ** Модели **
# ========================================================================

# ========================================================================
#### Линейная авторегрессионная модель

def gen_arf(ts, nar):
    return pd.DataFrame(
        array([ts.shift(i) for i in range(1,nar+1)]).T, 
        index = ts.index,
        columns = ['AR{:04}'.format(i) for i in range(1,nar+1)]
    )

class LAR:
    def __init__(self, endog, exog = None, nar = 1, regressor = LinearRegression()):
        self.endog = endog
        self.exog = exog
        if exog is not None: assert len(self.endog)==len(self.exog)
        self.nar = nar
        self.m = deepcopy(regressor)

    def fit(self):
        
        X = gen_arf(self.endog, self.nar)
        
        if self.exog is not None: X = hstack([X, self.exog]) 
        
        self.m.fit(X[self.nar:], self.endog[self.nar:])

        #self.coef_ = self.m.coef_
        #self.intercept_ = self.m.intercept_
        
        return self
    
    def predict(self, start = 0, end = -1, exog = None):
        # including end like in SARIMAX!

        n = len(self.endog)
        
        if start < self.nar: start = self.nar
        if end == -1: end = n - 1
        n_after = end - (n - 1) if end > n - 1 else 0

        if exog is not None: 
            assert self.exog is not None
            assert len(exog)==n_after
        
        y = self.endog.copy()
        
        dt = y.index[-1] - y.index[-2]
        y = y.append(pd.Series(index=pd.DatetimeIndex(
            start = y.index[-1] + dt, 
            freq = dt, 
            periods = n_after
        )))
        
        y_out = pd.Series(index=y.index)
        
        XE = vstack([self.exog, exog]) if exog is not None else self.exog
        if XE is None: XE = empty((n + n_after, 0))
            
        start_pred = start if start < n else n
        
        for i in range(start_pred, end+1):
            XA = array([flip( y[i-self.nar : i].values, 0)])
            X = hstack([XA, XE[i:i+1]])
            yy = self.m.predict(X)
            y_out[i] = yy
            if i>=n: y[i] = yy
        
        return y_out[start:end+1]
    
    def update(self, endog, exog=None):
        '''
        Appends endog and exog without re-fitting model. New data will be used for prognosis.
        Be sure time index is in coherence with existing, no checks inside.
        '''
        
        self.endog = pd.concat([self.endog, endog])
        if self.exog is not None:
            assert len(endog) == len(exog)
            assert exog is not None
            self.exog = pd.concat([self.exog, exog])
        
    def reset(self, tr):
        '''
        Cut updated endog and exog down to tr length 
        '''
        self.endog = self.endog[:tr]
        if self.exog is not None:
            self.exog = self.exog[:tr]
            
            
            
# ========================================================================
#### Линейная авторегрессия для совокупности рядов

'''
Напишем модель, состоящую из NHOURS = 6 линейных регрессоров для предсказания 6 отсчетов вперед от конца истории. Признаками (помимо внешних - гармоник и погоды) будут служить

- Отсчеты за K предыдущих часов
- Отсчеты за Kd предыдущих дней за тот же час
- Отсчеты за Kw предыдущих недель за тот же день и час

Кроме того, предусмотрим передачу в метод fit() модели (а именно там конструируются регрессоры) конкретного класса регрессора.

'''

def gen_arf2(df, K, Kd, Kw):
    return pd.concat(
        [df.shift(i)
         .rename(columns = {
             c:str(c)+'-%d'%i 
             for c in df.columns
         })
         for k,p in zip([K,Kd,Kw],[1,24,24*7]) 
         for i in range(p, p*(k+1),p)],
        axis=1
    )

class AGGLAR:

    def __init__(self, df, K=24, Kd=7, Kw=1, exog_xtra=None, *args, **kwargs):
        '''
        *args, **kwargs: model hyper parameters to generate exog, see gen_exog
        '''
        self.df = df
        self.nans = max(K, Kd*24, Kw*27*7) # most last tick, to cut NaNs from autoregression features 
        
        # generate exog features
        exog = gen_exog(df, *args, **kwargs)
        
        if exog_xtra is not None: exog = pd.concat([exog, exog_xtra], axis=1)

        # generate autoregression features
        arf = gen_arf2(df, K, Kd, Kw)
         
        self.X = pd.concat([exog, arf], axis=1)
        

    def fit(self, tr, Regressor = LinearRegression()):
        '''
        Fits regressors on tr length train set
        '''
        self.lrs = [
            deepcopy(Regressor)
            .fit(
                self.X[self.nans : tr-NHOURS], 
                self.df.shift(-i)[self.nans : (tr-NHOURS)]
            )
            for i in range(NHOURS)
        ]
            
        return self
        
    def predict(self, start, npred = 1):
        '''
        Predicts NHOURS ticks after given end_of_history = start - 1 npred times
        '''
        return swapaxes([
            self.lrs[i].predict(self.X[start:start+npred])
            for i in range(NHOURS)
        ],0,1)
        
    def Q(self, start, npred):
        '''
        Calculates prognosis error functional
        '''
        return sum([
            mean_absolute_error(
                self.lrs[i].predict(self.X[start:start+npred]),
                self.df[start+i:start+i+npred]
            )
            for i in range(NHOURS) 
        ]) / NHOURS

# ========================================================================

# ========================================================================

# ========================================================================
