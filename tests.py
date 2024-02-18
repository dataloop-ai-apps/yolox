from model_adapter import ModelAdapter
import dtlpy as dl


def test_predict(dataset: dl.Dataset, model: dl.Model):
    items_list = list()
    items_id = list()
    pages = dataset.items.list()
    for page in pages:
        for item in page:
            items_list.append(item)
            items_id.append(item.id)

    adapter = ModelAdapter(model_entity=model)
    adapter.predict_items(items_list)


def test_train(model: dl.Model):
    adapter = ModelAdapter(model_entity=model)
    adapter.train_model(model=model)


if __name__ == "__main__":
    dl.setenv('prod')
    project = dl.projects.get(project_name='Models-Intro')
    dataset = project.datasets.get(dataset_name='debugging-train-yolox')
    # model = project.models.get(model_name="new-yolox-debugger")
    model = project.models.get(model_name="yoloxM-debugger")

    test_train(model=model)

    test_predict(dataset=dataset, model=model)
