def tree_classification(training_set_features, testing_set_features, training_set_labels, testing_set_labels):
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    method = "decision_tree"
    scaler = StandardScaler()
    scaled_feats_train = scaler.fit_transform(training_set_features)
    svr = DecisionTreeClassifier(random_state=0)
    parameters = {'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}
    clf = GridSearchCV(svr, parameters, cv=5, scoring='accuracy')
    clf.fit(scaled_feats_train, training_set_labels)
    scaled_feats_test = scaler.transform(testing_set_features)
    predicted_lab_test = clf.predict(scaled_feats_test)
    best_score = clf.best_score_
    test_score = accuracy_score(testing_set_labels, predicted_lab_test, normalize=True)
    return method, best_score, test_score