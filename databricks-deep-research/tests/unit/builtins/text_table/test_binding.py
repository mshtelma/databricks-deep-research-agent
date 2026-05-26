from databricks_deep_research.tools.builtins.text_table.binding import (
    BindingInfo,
    BindingSource,
    RoleMap,
)


def test_role_map_required_fields():
    rm = RoleMap(id_column="chunk_id", content_column="content")
    assert rm.id_column == "chunk_id"
    assert rm.content_column == "content"
    assert rm.order_column is None


def test_binding_info_holds_role_map_and_metadata():
    rm = RoleMap(id_column="id", content_column="text")
    info = BindingInfo(
        name="treasury_chunks",
        fqn="cat.schema.tbl",
        description="desc",
        source=BindingSource.BOUND,
        roles=rm,
        numeric_columns=("count",),
        structured_passages={"table": "html"},
    )
    assert info.source is BindingSource.BOUND
    assert info.numeric_columns == ("count",)
    assert info.structured_passages == {"table": "html"}
