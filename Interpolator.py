'''
    Name:           Data Interpolator

    Description:    This takes 2D data and interpolates the points between to create finer resolution data using quadratic interpolation

    Date:           18 October 2022

    Author:         John Ferrier, NEU Physics

    Usage:          Currently, just initiate the class with the given data (1D numpy array each) and the amount of points wanted. (input points must be more than the input data size to work properly)
                        Inter   = Interpolator( x = input_x, y = input_y, points = 2000 )
                    After this, just call new_x and new_y for the interpolated data
                        x       = Inter.new_x
                        y       = Inter.new_y

    TO-DO:          Finish the integrators and then incorporate derivative function
'''
import numpy as np
import matplotlib.pyplot as plt

class Interpolator:

    # Initializes the class
    def __init__( self, 
                    x,                                                  # Independent Values 
                    y,                                                  # Dependent Values
                    points = 1000) -> None:                             # Points wanted
        
        self.x      = x                                                 # Independent Values
        self.y      = y                                                 # Dependnt Values
        self.p      = points                                            # Points wanted

        self.new_x  = np.linspace( self.x[ 0 ], self.x[ -1 ], self.p )  # New x-values to be returned
        self.new_y  = np.zeros_like( self.new_x )                       # New y-values to be returned
        self.int_y  = None                                              # Indefinite Integral of y(x)

        self.px     = len( x )                                          # x data count provided
        self.py     = len( y )                                          # y data count provided

        # Make sure there are enough points (needs at least 3 points)
        if self.p > 2:
            self._fit_data()
            self._interpolate_data()

    # Creates a matrix representation of interpolated data
    def _fit_data( self ):

        x_size      = 3*( self.px - 2 )
        self.mat_x  = np.zeros( ( x_size, x_size ), dtype = np.float32 )    # Matrix representation of x for interpolation
        self.mat_y  = np.zeros( ( x_size, 1 ), dtype = np.float32 )         # Matrix representation of y for interpolation
        self.c      = np.zeros( ( x_size, 1 ), dtype = np.float32 )         # Matrix representation of constants for interpolation

        for i in range( len( self.x ) - 2 ):

            # Build x Matrix
            self.mat_x[ 3*i ][ 3*i ]            = self.x[i]**2
            self.mat_x[ 3*i ][ 3*i + 1 ]        = self.x[i]
            self.mat_x[ 3*i ][ 3*i + 2 ]        = 1

            self.mat_x[ 3*i + 1 ][ 3*i ]        = self.x[i+1]**2
            self.mat_x[ 3*i + 1 ][ 3*i + 1 ]    = self.x[i+1]
            self.mat_x[ 3*i + 1 ][ 3*i + 2 ]    = 1

            self.mat_x[ 3*i + 2 ][ 3*i ]        = self.x[i+2]**2
            self.mat_x[ 3*i + 2 ][ 3*i + 1 ]    = self.x[i+2]
            self.mat_x[ 3*i + 2 ][ 3*i + 2 ]    = 1

            # Build y Matrix
            self.mat_y[ 3*i ]                   = self.y[ i ]
            self.mat_y[ 3*i + 1 ]               = self.y[ i + 1 ]
            self.mat_y[ 3*i + 2 ]               = self.y[ i + 2 ]             


        # Calculate Inverse X
        self.INV_mat_x  = np.linalg.inv( self.mat_x )
        self.c          = np.matmul( self.INV_mat_x, self.mat_y )

    # Extrapolates data from the original dataset and builds the newly requested data set
    def _interpolate_data( self ):

        # Figure which subsets fit
        self.fit_indices     = []

        for x in self.new_x:

            self.fit_indices.append( np.abs( self.x - x ).argmin() )

        for i in range( len( self.new_x ) ):

            # Adjust for the first and last indices
            if self.fit_indices[i] == 0:
                self.new_y[i]   = self.c[3*self.fit_indices[i]]*self.new_x[i]**2 + self.c[3*self.fit_indices[i]+1]*self.new_x[i] + self.c[3*self.fit_indices[i]+2]
            elif self.fit_indices[i] == self.fit_indices[-1]:
                self.new_y[i]   = self.c[ 3*( self.fit_indices[ i ] - 2 ) ]*self.new_x[ i ]**2 + self.c[ 3*( self.fit_indices[ i ] - 2 ) + 1 ]*self.new_x[i] + self.c[ 3*( self.fit_indices[ i ] - 2 ) + 2 ]
            else:
                self.new_y[i]   = self.c[ 3*( self.fit_indices[ i ] - 1 ) ]*self.new_x[ i ]**2 + self.c[ 3*( self.fit_indices[ i ] - 1 ) + 1 ]*self.new_x[i] + self.c[ 3*( self.fit_indices[ i ] - 1 ) + 2 ]

    # TODO Calculates the integral from the interpolated equations
    def definite_integral( self, start_x = None, end_x = None ):

        if start_x is not None:

            # New x index
            xstrt_idx   = np.abs( self.new_x - start_x ).argmin()

            # Old x index
            oxstrt_idx  = np.abs( self.x - start_x ).argmin()
            l_ox        = len( self.x )                             # Length of old x

            # Ensure not at the end
            if oxstrt_idx > ( l_ox - 3 ):
                cstrt_idx   = 3*( oxstrt_idx - 2 )
            else:
                cstrt_idx   = 3*oxstrt_idx

        else:
            xstrt_idx   = 0
            cstrt_idx   = 0

        # Look at the end points
        if end_x is not None:

            # New x index
            xend_idx    = np.abs( self.new_x - end_x ).argmin()

            # Old x index
            oxend_idx   = np.abs( self.x - end_x ).argmin()
            l_ox        = len( self.x )                             # Length of old x

            # Ensure not at the end
            if oxend_idx > ( l_ox - 3 ):
                cend_idx    = 3*( oxend_idx - 2 )
            else:
                cend_idx    = 3*oxend_idx

        else:
            xend_idx    = -1
            cend_idx    = -3

        integ   = ( ( self.c[ cend_idx ]/3.)*self.new_x[ xend_idx ]**3 + (self.c[ cend_idx + 1 ]/2.)*self.new_x[ xend_idx ]**2 + self.c[ cend_idx + 2 ]*self.new_x[ xend_idx ] )
        print( f"integ_1 = {integ}" )
        integ2  = ( ( self.c[ cstrt_idx ]/3.)*self.new_x[ xstrt_idx ]**3 + (self.c[ cstrt_idx + 1 ]/2.)*self.new_x[ xstrt_idx ]**2 + self.c[ cstrt_idx + 2 ]*self.new_x[ xstrt_idx ] )
        print( f"integ_2 = {integ2}" )

        return integ - integ2

    # TODO Generates the function for the indefinite integral of the given data
    def indefinite_integral( self ):

        self.int_y  = np.zeros_like( self.new_x )
        self.D      = np.zeros_like( self.int_y )

        # Need to refit the new function in order to avoid issues with constants



        for i in range( len( self.new_x ) ):

            # If the first indices (Fit to the first quadratic)
            if self.fit_indices[i] == 0:

                self.D[i]       = 0.                 

                self.int_y[i]   = (1./3.)*self.c[ 3*self.fit_indices[i] ]*self.new_x[ i ]**3 
                self.int_y[i]   += (1./2.)*self.c[ 3*self.fit_indices[i] + 1 ]*self.new_x[ i ]**2 
                self.int_y[i]   += self.c[ 3*self.fit_indices[i] + 2 ]*self.new_x[ i ] + self.D[i]

            # If the last elements (Fit to the mid quadratic )
            elif self.fit_indices[i] == self.fit_indices[-1]:
                
                self.D[i]       = (1./3.)*( self.c[ 3*( self.fit_indices[ i - 1 ] - 2 ) ] - self.c[ 3*( self.fit_indices[ i ] - 2 ) ] )*self.new_x[ i ]**3 
                self.D[i]       += (1./2.)*( self.c[ 3*( self.fit_indices[ i - 1 ] - 2 ) + 1 ] - self.c[ 3*( self.fit_indices[ i ] - 2 ) + 1 ] )*self.new_x[i]**2 
                self.D[i]       += ( self.c[ 3*( self.fit_indices[ i - 1 ] - 2 ) + 2 ] - self.c[ 3*( self.fit_indices[ i ] - 2 ) + 2 ] )*self.new_x[i] + self.D[ i - 1 ]

                self.int_y[i]   = (1./3.)*self.c[ 3*( self.fit_indices[ i ] - 2 ) ]*self.new_x[ i ]**3 + (1./2.)*self.c[ 3*( self.fit_indices[ i ] - 2 ) + 1 ]*self.new_x[i]**2 + self.c[ 3*( self.fit_indices[ i ] - 2 ) + 2 ]*self.new_x[i] + self.D[i]
            
            # Everything in the middle (Fit to the last quadratic)
            else:
                
                self.D[i]       = ( 1./3. )*( self.c[ 3*( self.fit_indices[ i - 1 ] - 1 ) ] - self.c[ 3*( self.fit_indices[ i ] - 1 ) ] )*self.new_x[ i ]**3 
                self.D[i]       += ( 1./2. )*( self.c[ 3*( self.fit_indices[ i - 1 ] - 1 ) + 1 ] - self.c[ 3*( self.fit_indices[ i ] - 1 ) + 1 ] )*self.new_x[i]**2 
                self.D[i]       += ( self.c[ 3*( self.fit_indices[ i - 1 ] - 1 ) + 2 ] - self.c[ 3*( self.fit_indices[ i ] - 1 ) + 2 ] )*self.new_x[i] + self.D[ i - 1 ]

                self.int_y[i]   = ( 1./3. )*self.c[ 3*( self.fit_indices[ i ] - 1 ) ]*self.new_x[ i ]**3 
                self.int_y[i]   += ( 1./2. )*self.c[ 3*( self.fit_indices[ i ] - 1 ) + 1 ]*self.new_x[i]**2 
                self.int_y[i]   += self.c[ 3*( self.fit_indices[ i ] - 1 ) + 2 ]*self.new_x[i] + self.D[ i ]

        return self.int_y

    # TODO Generates the function for the derivate of the given data
    def indefinite_derivative( self ):
        pass

    # TODO Returns the derivative value at a point
    def derivative( self, x_point = 0 ):
        pass

    # Plots the Original data set against the new dataset for a visual representation of the extrapolated data
    def plot_Old_v_New( self ):

        plt.plot( self.x, self.y, 'ro' )
        plt.plot( self.new_x, self.new_y, 'b-' )
        if self.int_y is not None:
            plt.plot( self.new_x, self.int_y, 'g-' )

        plt.show()

if __name__ == "__main__":

    test_x  = np.linspace( -2*np.pi, 2*np.pi, 50 )
    test_y  = np.sin( test_x )
    Inter   = Interpolator( x = test_x, y = test_y, points = 2000 )

    # Plot the output for comparison purposes
    Inter.plot_Old_v_New()
    