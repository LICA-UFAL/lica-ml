(ns knn.core)


(defn most-common-class
  [classes]
  (->> classes
    frequencies
    (sort-by val)
    last
    first))

(defn get-classes
  [instances]
  (map #(get % 1) instances))

(defn get-features
  [instances]
  (get instances 0))

(defn sort-fn
  [sim-fn sample data]
  (->> data
    get-features
    (sim-fn sample)))

(defn closer-instances
  [classifier sample data]
  (->> data
    (sort-by
      (partial sort-fn (classifier :similarity-function) sample))
    reverse
    (take (classifier :k))))

(defn predict-sample
  [classifier data sample]
  (->> data
    (closer-instances classifier sample)
    get-classes
    most-common-class))

(defn transform-entry
  [x-data y-data]
  (map (fn [x y] [x y]) x-data y-data))

(defn predict
  [classifier x-data y-data sample]
  (predict-sample classifier (transform-entry x-data y-data) sample))
