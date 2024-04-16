def calculate_rca_by_country(data, country_column, commodity_column):
    
    world_export_proportion = data[commodity_column].sum() / data[commodity_column].count()

    country_groups = data[[country_column, commodity_column]].groupby(country_column)
    rca = (country_groups.sum() / country_groups.count()) / world_export_proportion

    return rca