def anomalies( fdat, yr1=None, yr1clim=None, yr2clim=None, nperyr=None ):
    '''Calculate anomalies and climatology of nperyr ( number of records per year ) data.

    Inputs
        fdat(time,space)
        yr1     first year of data (optional, if =None, calculates monthly anomalies of all data)
        yr1clim, yr2clim (optional, first and last years to use for climatology)
        nperyr (optional, default = 12 for monthly data )
    Outputs  anom, clim
        anom    anomaly dataset
        clim      climatology

    This is the python version of my Matlab script.  "Broadcasting" code by Jeremy McGibbon
    
    Todd Mitchell, February 2019'''

    import sys
    import numpy as np
    
    if nperyr is None:
        nperyr = 12
    
    nt = fdat.shape[ 0 ]
    nxyz = fdat.shape[ 1: ]
    if yr1 is None:  # Assumes monthly data, climatology of entire dataset
        yr1 = 1
        yr1clim = 1
        yr2clim = int( nt/12 )
        if nt % 12 != 0:
            print( 'Algorithm expects whole years of monthly data.' )
        skip = 0    # skip and nt2 are used for specifying climatologies of a subset of years
        nt2  = nt
    else:
        skip = ( yr1clim - yr1 ) * nperyr
        nt2  = ( yr2clim - yr1clim + 1 ) * nperyr

    print( 'yr1, yr1clim, yr2clim, skip, nt2\n', yr1, yr1clim, yr2clim, skip, nt2 )
    
    clim = np.zeros( (nperyr,) + fdat.shape[ 1: ] )
    anom = np.zeros( fdat.shape )

    "The number of dimensions is {}".format( fdat.ndim ) 
    if fdat.ndim<=2:
        for icnt in np.arange( nperyr ):
            clim[icnt] = np.nanmean( fdat[ skip+icnt:skip+nt2:nperyr ] )
            anom[ icnt::nperyr ] = fdat[ icnt::nperyr ] - ( clim[icnt] )[None]

    else:
        "Need to add the code to handle {} dimensions.  It's easy!".format( fdat.ndim )
# seasonal_cycle[i,:,:] = np.mean(sst[i::12,:,:])
# sst_anom[i::12,:,:] = sst[i::12,:,:] - (seasonal_cycle[i,:,:])[None,:,:]  is the form
# Broadcasting rules are described at
# See https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html

    return anom, clim
def arclength( lat1, lon1, lat2, lon2, radius=None ):
    """Arc-length distance in km

    Assumes angles in degrees ( not radians ). 

    Todd Mitchell, April 2019"""

    if radius is None:
        radius = 6.37e3  # in km

    import numpy as np
    meanlat = np.mean( ( lat1, lat2 ) )
    rcosine   = radius * np.cos( np.deg2rad( meanlat ) )
    a = rcosine * ( lon2 - lon1 ) / 360  * 2 * np.pi
    b = radius  * ( lat2  - lat1 )   / 360 * 2 * np.pi

    return np.sqrt( a * a + b * b )

def fill_year( array, nperyr=None ):
    '''Append missing values ("NaN"s) to a time series / data file
    to make complete years of data.

    Input: 
    array      a time series or data matrix where time varies by row
    nperyer    (optional) number of records in a year; default 12

    Example: tpm.fill_year( ts )

     Todd Mitchell, June 2018 '''
    import numpy as np
    if nperyr is None:
        nperyr = 12
    nperyr *= 1.0
    if len( array.shape )==1: 
        nt = array.shape
        nt = nt[0]
        nt2 = np.ceil( nt/nperyr ) * nperyr
        nt2 = int( nt2 )
        temp = np.ones( ( nt2-nt ), dtype="int" ) * np.nan
        array = np.hstack( ( array, temp ) )
    else: 
        nt, nx = array.shape
        nt2 = np.ceil( nt/nperyr ) * nperyr
        nt2 = int( nt2 )
        temp = np.ones( ( nt2-nt, nx ), dtype="int" ) * np.nan
        array = np.vstack( ( array, temp ) )
    return( array )
def find_latlon( xgrid, ygrid, lat, lon ):
    '''Identify the x and y gridpoints that are closest to 
    the input lat and lon.  

    The code was written for Huancayo Peru, which is at a longitude of -75.21 
    and the input grid spans 275 to 330.  Be careful that the answer you 
    get makes sense ! 

    Huancayo -12.0668 latitude and -75.2103 longitude

    The algorithm assumes that xgrid, ygrid are vectors.

    The algorithm flags if 2 longitude or 2 latitude points are returnd.

    Todd Mitchell, January 2019'''

    import numpy as np

    temp = abs( ygrid - lat )
    yval = np.where( temp == min(temp) )
    if len(yval) > 1:
        print( len(yval), ' gridpoints nearest to ', lat )

    if lon<0:
        lon = lon + 360

    temp = abs( xgrid - lon )
    xval = np.where( temp == min(temp) )
    if len(xval) > 1: 
        print( len(xval), ' gridpoints nearest to ', lon )

    return{ 'yval': yval, 'xval': xval }
def plot_vertical_lines( xvals, yvals, zorder=None, color=None ):
    """Plot vertical lines on plots.  The default is that these lines are very light gray.
    zorder is used to force these lines to be beneath the other plot elements

Input:
    xvals   tuple (of first, last xtick values, and increment ) or numpy.ndarray of values
    yvals   tuple minimum and maximum y-value to plot
    zorder  default is 1
    color   default is ( 0.8, 0.8 0.8 ), which is light gray
    
    https://matplotlib.org/3.1.1/gallery/misc/zorder_demo.html   Layering of plot elements
    
    Todd Mitchell, September 2019"""
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if zorder is None:
        zorder = 1
    if color is None:
        color = ( 0.9, 0.9, 0.9 )

    if type( xvals ) is tuple:
        for icnt, xval in enumerate( np.arange( xvals[0], xvals[1]+xvals[2], xvals[2] ) ):
            plt.plot( ( xval, xval ), ( yvals[0], yvals[1] ), zorder=zorder, color=color )
    else:  # it is a numpy.ndarray
        for icnt, xval in enumerate( xvals ):
            plt.plot( ( xval, xval ), ( yvals[0], yvals[1] ), zorder=zorder, color=color )
            
    return
    
def space_longitudes():
    '''Adjust the aspect ratio of maps to take into account the 
    convergence of map meridians.
     Todd Mitchell, October 2018 '''
    import matplotlib.pyplot as plt
    import numpy as np
    xl, xr = plt.xlim()   # yields the x limits
    yb, yt = plt.ylim()
    plt.axes().set_aspect( aspect=( xr - xl ) / ( ( yt - yb ) * np.cos( np.radians( np.mean( [ yt, yb ] ) ) ) ) )
    return()
def threetotwo( array ):
    '''Turns a 3-dimensional array( nx, ny, nt ) into a 
    2-dimensional array ( nt, nx*ny ).

    Todd Mitchell, November 2018.'''
    import numpy as np
    nt, ny, nx = array.shape
    if min( [ nt, ny, nx ] )==1:
        print( 'This is a 2-dimensional array.' )
    array = np.transpose( array )
    array = np.reshape( array, ( nx*ny, nt ) )
    array = array.T
    return( array )
def time_shift( fdat, yrfst1, yrlst1, yrfst2, yrlst2, nperyr=None ):
    '''Change the period of record of a dataset.  If necessary, put in NaNs to
    make the record longer.

    Input variables:
    fdat                data -- the zeroeth dimension is time.
    yrfst1, yrlst1      first and last years of the input  data.
    yrfst2, yrlst2      first and last years of the output data.
    nperyr              (optional) number of records per year.  default = 12

    Todd Mitchell, December 2020'''

    import numpy as np
    
    if nperyr is None:
        nperyr = 12    # "monthly" is the default option 
    
    nt = ( yrlst1 - yrfst1 + 1 ) * nperyr
    if nt != fdat.shape[0]:
        f'The specified first and last years are inconsistent with the time series length.'
        f'fdat.shape {fdat.shape} nt {nt}'

# Remove year(s) from the beginning of the series/data
    if yrfst2 > yrfst1:
        fdat = fdat[(yrfst2-yrfst1)*nperyr:]

# Remove year(s) from the end of the series/data
    if yrlst2 < yrlst1:
        fdat = fdat[:-(yrlst1-yrlst2)*nperyr]

# Prepend NaNs to make the series/data longer
    if yrfst2 < yrfst1:
        nfill = ( yrfst1 - yrfst2 ) * nperyr
        if fdat.ndim==1:
            fdat = np.concatenate( ( np.zeros(nfill)*np.nan, fdat ), axis=0 )
        else:
            fdat = np.concatenate( ( np.zeros((nfill,*fdat.shape[1:][:]))*np.nan, fdat ), axis=0 )
        
# Append NaNs to make the series/data longer
    if yrlst2 > yrlst1:
        nfill2 = ( yrlst2 - yrlst1 ) * nperyr
        if fdat.ndim==1:
            fdat = np.concatenate( ( fdat, np.zeros(nfill2)*np.nan ), axis=0 )
        else:
            fdat = np.concatenate( ( fdat, np.zeros((nfill2,*fdat.shape[1:][:]))*np.nan ), axis=0 )

    return fdat

def write_ts( ts, yr1, yr2, yrfst=None ):
    '''write_ts( ts, yr1, yr2 ) writes a monthly timeseries in table form to stdio.

    This is the beginning of converting write_ts.m to python.

     Todd Mitchell, February 2019 '''

    import numpy as np
    import sys
    
    if yrfst is None:
        yrfst = yr1

    nyr = yr2 - yr1 + 1
    
    print( 'ts.shape yields', ts.shape )

    a = np.reshape( np.round( ts*10 ), ( nyr, 12 ) )
    a[ np.isnan(a) ] = -999
    b = np.arange( yr1, yr2+1 )
    b = np.expand_dims( b, axis=1 )
    b = np.concatenate( ( b, a ), axis=1 ).astype(int)
    np.savetxt( sys.stdout, b, fmt='%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d' )
def yearsmonths( yr1, yr2=None ):
    '''yearsmonths(yr1, yr2 ) returns a dictionary of monthly values of years,
    and calendar month for the input years

     yr2 optional, default is yr2 = yr1

     Output ã dictionary of monthly values of 
     years           year
     months          calendar month

     Example: a = tpm.yearsmonthsdays( 1970 ), a[ 'years' ] = 1970, 1970, ...

     Todd Mitchell, January 2019 '''

    import numpy as np
    if yr2 is None:
        yr2 = yr1

    nyr = yr2 - yr1 + 1

    years = np.arange( yr1, yr2+1 )
    years = np.expand_dims( years, axis=1 )
    ones = np.ones( ( 1, 12 ) )
    years = years * ones
    years = np.reshape( years, ( int(nyr)*12, 1 ) )

    months = np.arange( 1, 12+1 )
    months = np.expand_dims( months, axis=1 )
    ones = np.ones( ( 1, int(nyr) ) )
    months = ( months * ones ).T
    months = np.reshape( months, ( int(nyr)*12, 1 ) )

    return { 'years': years, 'months':months }
def yearsmonthsdays( yr1, yr2=None ):
    '''yearsmonthsdays(yr1, yr2 ) returns a dictionary of daily values of year,
    month, day of month, and Julian day for the input year(s).

    The function handles Leap Years.  Leap Years take into account
    that the Earth's period of orbit around the Sun is 365.24... days.  
    Leap Years have an additional day compared to regular years, and that day
    is appended to February for a total of 366 days in that year.  Years evenly
    divisible by 4 are Leap Years.  This is modified for century years:
    Years evenly divisible by 100 are not Leap Years, unless, of course,
    they are evenly divisible by 400.  The years 2000 and 1900 are and are not
    Leap Years, respectively. 

     yr2 optional, default is yr2 = yr1

     Output ã dictionary of daily values of 
     years           year
     months          calendar month
     days            day of month
     jdays           Julian Day ( 1 - 365 ) or ( 1 - 366 for Leap Years )

     Example: a = tpm.yearsmonthsdays( 1970 ), a[ 'years' ] = 1970, 1970, ...

     Todd Mitchell, June 2022 '''
    import numpy as np
    if yr2 is None:
        yr2 = yr1
    ndays = np.array( [    # leap and 3 non-leap years, number of days in each calendar month, 
        [ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ],
        [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ], 
        [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ], 
        [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ] ] )
    for year in np.arange( yr1, yr2+1 ):
#       The variable "a" will be used as an index for the "ndays" array
        a = year % 4    # Is the year a leap year?  a = 0 for a leap year
        if ( year%100 ) == 0:
            a = 1
        if ( year%400 ) == 0:
            a = 0
        for month in np.arange( 12 ):
            yearstemp  = np.ones( ( ndays[a,month],1 ), dtype="int" ) * year
            monthstemp = np.ones( ( ndays[a,month],1 ), dtype="int" ) * (month+1)
            daystemp   = np.arange( ndays[a,month] ).reshape( ( -1, 1 ) ) + 1
#           print( "year, month", year, month )
            if month==0 and year==yr1:
#               print( "Inside the month==0/year==yr1 branch." )
                years = yearstemp
                months = monthstemp
                days = daystemp
#               print( "Inside the month==0/year==yr1 branch.  days.shape", days.shape )
            else:
#               print( "Inside the else branch.  days.shape, daystemp.shape", days.shape, daystemp.shape )
                years = np.vstack( ( years, yearstemp ) )
                months = np.vstack( ( months, monthstemp ) )
                days = np.vstack( ( days, daystemp ) )
#               print( "Inside the else branch.  days.shape", days.shape )
#       print( "After the if/else code.  days.shape", days.shape )
        jdaystemp = np.arange(366)+1
        if a>0: jdaystemp = np.arange(365)+1
        jdaystemp = jdaystemp.reshape( ( -1, 1 ) )
        if year==yr1:
            jdays = jdaystemp
        else:
            jdays = np.vstack( ( jdays, jdaystemp ) )
        days  = days.flatten()
        jdays = jdays.flatten()
        days  = days.reshape( (-1, 1 ) )
        jdays = jdays.reshape( (-1, 1 ) )
#    return ( years, months, days, jdays )
    return { 'years': years, 'months':months, 'days':days, 'jdays':jdays }
