(ns knn.similarity-functions-test
  (:require [clojure.test :refer :all]
            [knn.similarity-functions :refer :all]))

(def first-instance [1 2 3])
(def second-instance [1 2 3])

(deftest euclidean-distance-tests
  (testing "should return distance equals 0"
    (is (= 0.0 (euclidean-distance first-instance second-instance)))))
