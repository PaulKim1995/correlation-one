# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import logging

import pandas as pd


def reformat_prices(df, columns):
    """
    Deals with the $ sign in price columns
    Args:
        df ():
        columns ():

    Returns:

    """
    # Check
    assert isinstance(columns, list), "{} is not a list".format(columns)
    # Iterate
    for column in columns:
        df[column] = df[column].replace('[\$,)]', '', regex=True).astype(float)
    return df


def reformat_booleans(df, columns):
    # Check
    assert isinstance(columns, list), "{} is not a list".format(columns)
    # Iterate
    for column in columns:
        df[column] = df[column].apply(lambda x: True if x == 't' else False)
    return df


def expand_list_in_cell(df, column_to_expand):
    """
    Expansion
    Args:
        df ():
        column_to_expand ():

    Returns:

    """
    # Removing dict structure  and perform cleaning
    column_to_expand_values = list(df[column_to_expand].values)
    if any("{" in x for x in column_to_expand_values) \
            or any("{" in x for x in column_to_expand_values) \
            or any("{" in x for x in column_to_expand_values):
        df[column_to_expand] = df[column_to_expand].apply(
            lambda x: x.replace("\"", "").replace("{", "").replace("}", ""))
    # Generate the list by splitting
    df[column_to_expand] = df[column_to_expand].apply(lambda x: x.split(','))
    # Lower case
    df[column_to_expand] = df[column_to_expand].apply(lambda x: [amenity.lower().replace(' ', '_') for amenity in x])

    # List all tags
    all_tags = set()
    for index, value in df[column_to_expand].iteritems():
        all_tags = all_tags.union(set(value))
    # Cleaning
    all_tags.remove('')
    if 'translation missing: en.hosting_amenity_49' in all_tags:
        all_tags.remove('translation missing: en.hosting_amenity_49')
    if 'translation missing: en.hosting_amenity_50' in all_tags:
        all_tags.remove('translation missing: en.hosting_amenity_50')
    # Logging
    logging.warning("You have expanded a column of {} tags".format(len(all_tags)))

    for tag in sorted(all_tags):
        df[tag] = df[column_to_expand].apply(lambda x: 1 if tag in x else 0)
    df.drop([column_to_expand], inplace=True, axis=1)

    return df, all_tags


def load_listings(loading_path='data/listings.csv'):
    """
    Wrap-up function
    Args:
        loading_path ():

    Returns:

    """
    listings_df = pd.read_csv(loading_path)
    listings_df = reformat_prices(listings_df, ['price'])
    listings_df = reformat_booleans(listings_df, ['has_availability', 'instant_bookable'])
    listings_df, all_amenities = expand_list_in_cell(listings_df, 'amenities')
    return listings_df, all_amenities


if __name__ == '__main__':
    # For testing purposes
    listings_df_ = load_listings()
    print('main is over')
