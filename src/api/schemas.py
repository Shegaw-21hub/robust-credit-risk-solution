from pydantic import BaseModel

class CreditRiskRequest(BaseModel):
    num__Recency: float
    num__Frequency: float
    num__MonetarySum: float
    num__MonetaryMean: float
    num__MonetaryStd: float
    cat__ProductCategory_airtime: int
    cat__ProductCategory_data_bundles: int
    cat__ProductCategory_financial_services: int
    cat__ProductCategory_movies: int
    cat__ProductCategory_other: int
    cat__ProductCategory_ticket: int
    cat__ProductCategory_transport: int
    cat__ProductCategory_tv: int
    cat__ProductCategory_utility_bill: int
    cat__ChannelId_ChannelId_1: int
    cat__ChannelId_ChannelId_2: int
    cat__ChannelId_ChannelId_3: int
    cat__ChannelId_ChannelId_5: int
    cat__PricingStrategy_0: int
    cat__PricingStrategy_1: int
    cat__PricingStrategy_2: int
    cat__PricingStrategy_4: int
    temp__TransactionHour: float
    temp__TransactionDay: float
    temp__TransactionMonth: float
