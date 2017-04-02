#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    errors = sorted( [ ( ages[x][0], net_worths[x][0], abs( net_worths[x][0]-predictions[x][0] ) ) for x in range( len( predictions ) ) ], key=lambda error: error[2] )
    cleaned_data = errors[:len(errors)/10*9]
    return cleaned_data

