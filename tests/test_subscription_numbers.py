from services.subscription_numbers import subscription_multiple


def test_decimal_commas_are_not_hundredfold_demand():
    assert subscription_multiple('6,46 x')==6.46
    assert subscription_multiple('(0,25)')==.25
    assert subscription_multiple('6.46x')==6.46
    assert subscription_multiple('1,234.56x')==1234.56
    assert subscription_multiple('1,23,456')==123456
    assert subscription_multiple('Not published') is None
    assert subscription_multiple('-1.5') is None
    assert subscription_multiple('1,2,3') is None
