# def pytest_collection_modifyitems(session, items):
#     #for item in items:
#         #print('fixtures', session._fixturemanager.getfixtureinfo(item, item.function, None))
#     #parents = []
#     for item in set(item.parent for item in items):
#         print('Parent fixtures', session._fixturemanager.getfixtureinfo(item, item.function, None))
# 
