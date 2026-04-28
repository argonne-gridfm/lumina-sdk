from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

"""
Data sources differ in feature ordering, so instead of hardcoding the feature order
we define the following schemas in pydantic and then align everything within the OPFDataset class during preprocessing
 
The canonical feature ordering is the following JSON schema from pglib-opf,
hdf5 acopf data from exagrid needs to be aligned.

Complete schema documentation is available at: 
https://github.com/argonne-gridfm/GridAI-documentation/blob/main/data_generation/opfdata_schema_documentation.md

"""


class OPFSchemaModel(BaseModel):
    """Base Pydantic model defining a column-ordered feature schema for OPF data.

    Subclasses declare fields whose order defines the canonical column layout
    of the corresponding numpy feature array. Provides utilities for schema
    introspection, cross-schema alignment, and numpy conversion.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return the ordered list of feature field names.

        Returns:
            List[str]: Field names in declaration order.
        """
        return list(cls.model_fields.keys())

    @classmethod
    def get_field_indices(cls) -> Dict[str, int]:
        """Return a mapping from field names to their column indices.

        Returns:
            Dict[str, int]: ``{field_name: column_index}`` for every field.
        """
        return {name: i for i, name in enumerate(cls.get_feature_names())}

    @classmethod
    def get_alignment_map(cls, other: type["OPFSchemaModel"]) -> Dict[int, int]:
        """Compute a column index mapping from this schema to another.

        Only fields present in both schemas are included.

        Args:
            other (type[OPFSchemaModel]): Target schema class.

        Returns:
            Dict[int, int]: ``{source_column: target_column}`` for shared fields.
        """
        cls_indices = cls.get_field_indices()
        other_indices = other.get_field_indices()

        mapping = {}
        for name, idx in cls_indices.items():
            if name in other_indices:
                mapping[idx] = other_indices[name]
        return mapping

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> "OPFSchemaModel":
        """Create a model instance from a 1-D numpy array of feature values.

        Args:
            data (np.ndarray): Array whose elements correspond to schema fields
                in declaration order.

        Returns:
            OPFSchemaModel: Populated model instance.
        """
        return cls(**dict(zip(cls.get_feature_names(), data.tolist())))

    def to_numpy(self) -> np.ndarray:
        """Convert this model instance to a 1-D numpy array of feature values.

        Returns:
            np.ndarray: Feature values in declaration order.
        """
        return np.array([getattr(self, field) for field in self.get_feature_names()])


class JSONBus(OPFSchemaModel):
    """Bus features in JSON format."""
    base_kv: float = Field(..., description="Base voltage (kV)")
    bus_type: int = Field(..., description="Bus type (1=PQ, 2=PV, 3=Ref, 4=Isolated)")
    vmin: float = Field(..., description="Minimum voltage magnitude (p.u.)")
    vmax: float = Field(..., description="Maximum voltage magnitude (p.u.)")

class JSONGenerator(OPFSchemaModel):
    """Generator features in JSON format."""
    mbase: float = Field(..., description="Machine base power (MVA)")
    pg: float = Field(..., description="Active power generation (p.u.)")
    pmin: float = Field(..., description="Minimum active power output (p.u.)")
    pmax: float = Field(..., description="Maximum active power output (p.u.)")
    qg: float = Field(..., description="Reactive power generation (p.u.)")
    qmin: float = Field(..., description="Minimum reactive power output (p.u.)")
    qmax: float = Field(..., description="Maximum reactive power output (p.u.)")
    vg: float = Field(..., description="Voltage setpoint (p.u.)")
    cost_c2: float = Field(..., description="Quadratic cost coefficient ($/MW²/h)")
    cost_c1: float = Field(..., description="Linear cost coefficient ($/MW/h)")
    cost_c0: float = Field(..., description="Constant cost coefficient ($/h)")

class JSONLoad(OPFSchemaModel):
    """Load features in JSON format."""
    pd: float = Field(..., description="Active power demand (p.u.)")
    qd: float = Field(..., description="Reactive power demand (p.u.)")

class JSONShunt(OPFSchemaModel):
    """Shunt features in JSON format."""
    bs: float = Field(..., description="Shunt susceptance (p.u.)")
    gs: float = Field(..., description="Shunt conductance (p.u.)")

class JSONACLine(OPFSchemaModel):
    """AC Line features in JSON format."""
    angmin: float = Field(..., description="Minimum voltage angle difference (radians)")
    angmax: float = Field(..., description="Maximum voltage angle difference (radians)")
    b_fr: float = Field(..., description="Charging susceptance (from side) (p.u.)")
    b_to: float = Field(..., description="Charging susceptance (to side) (p.u.)")
    br_r: float = Field(..., description="Series resistance (p.u.)")
    br_x: float = Field(..., description="Series reactance (p.u.)")
    rate_a: float = Field(..., description="Long-term thermal rating (MVA)")
    rate_b: float = Field(..., description="Short-term thermal rating (MVA)")
    rate_c: float = Field(..., description="Emergency thermal rating (MVA)")

class JSONTransformer(OPFSchemaModel):
    """Transformer features in JSON format."""
    angmin: float = Field(..., description="Minimum voltage angle difference (radians)")
    angmax: float = Field(..., description="Maximum voltage angle difference (radians)")
    br_r: float = Field(..., description="Series resistance (p.u.)")
    br_x: float = Field(..., description="Series reactance (p.u.)")
    rate_a: float = Field(..., description="Long-term thermal rating (MVA)")
    rate_b: float = Field(..., description="Short-term thermal rating (MVA)")
    rate_c: float = Field(..., description="Emergency thermal rating (MVA)")
    tap: float = Field(..., description="Transformer tap ratio (p.u.)")
    shift: float = Field(..., description="Phase shift angle (radians)")
    b_fr: float = Field(..., description="Charging susceptance (from side) (p.u.)")
    b_to: float = Field(..., description="Charging susceptance (to side) (p.u.)")

class JSONBusSolution(OPFSchemaModel):
    """Bus solution features in JSON format."""
    va: float = Field(..., description="Voltage angle (radians)")
    vm: float = Field(..., description="Voltage magnitude (p.u.)")

class JSONGeneratorSolution(OPFSchemaModel):
    """Generator solution features in JSON format."""
    pg: float = Field(..., description="Active power generation (p.u.)")
    qg: float = Field(..., description="Reactive power generation (p.u.)")

class JSONEdgeSolution(OPFSchemaModel):
    """Edge solution (AC Line/Transformer) features in JSON format."""
    pt: float = Field(..., description="Active power flow (to side) (p.u.)")
    qt: float = Field(..., description="Reactive power flow (to side) (p.u.)")
    pf: float = Field(..., description="Active power flow (from side) (p.u.)")
    qf: float = Field(..., description="Reactive power flow (from side) (p.u.)")

class H5Bus(OPFSchemaModel):
    """Bus features in HDF5 format."""
    vmin: float = Field(..., description="Minimum voltage magnitude (p.u.)")
    vmax: float = Field(..., description="Maximum voltage magnitude (p.u.)")
    zone: Optional[float] = Field(None, description="Zone identifier")
    area: Optional[float] = Field(None, description="Area identifier")
    bus_type: float = Field(..., description="Bus type (1=PQ, 2=PV, 3=Ref)")

class H5Generator(OPFSchemaModel):
    """Generator features in HDF5 format."""
    pmax: float = Field(..., description="Maximum active power output (p.u.)")
    pmin: float = Field(..., description="Minimum active power output (p.u.)")
    qmax: float = Field(..., description="Maximum reactive power output (p.u.)")
    qmin: float = Field(..., description="Minimum reactive power output (p.u.)")
    cost_c2: float = Field(..., description="Quadratic cost coefficient ($/MW²/h)")
    cost_c1: float = Field(..., description="Linear cost coefficient ($/MW/h)")
    cost_c0: float = Field(..., description="Constant cost coefficient ($/h)")
    vg: float = Field(..., description="Voltage setpoint (p.u.)")
    mbase: float = Field(..., description="Machine base power (MVA)")
    gen_status: float = Field(..., description="Generator status (1=on, 0=off)")

class H5Load(OPFSchemaModel):
    """Load features in HDF5 format."""
    pd: float = Field(..., description="Active power demand (p.u.)")
    qd: float = Field(..., description="Reactive power demand (p.u.)")

class H5Shunt(OPFSchemaModel):
    """Shunt features in HDF5 format."""
    gs: float = Field(..., description="Shunt conductance (p.u.)")
    bs: float = Field(..., description="Shunt susceptance (p.u.)")

class H5ACLine(OPFSchemaModel):
    """AC Line features in HDF5 format."""
    angmin: float = Field(..., description="Minimum voltage angle difference (radians)")
    angmax: float = Field(..., description="Maximum voltage angle difference (radians)")
    br_r: float = Field(..., description="Series resistance (p.u.)")
    br_x: float = Field(..., description="Series reactance (p.u.)")
    b_fr: float = Field(..., description="Half of br_b (p.u.)")
    b_to: float = Field(..., description="Half of br_b (p.u.)")
    rate_a: float = Field(..., description="Long-term thermal rating (MVA)")
    rate_b: float = Field(..., description="Short-term thermal rating (MVA)")
    rate_c: float = Field(..., description="Emergency thermal rating (MVA)")
    br_status: Optional[float] = Field(None, description="Branch status (1=in-service, 0=out)")

class H5Transformer(OPFSchemaModel):
    """Transformer features in HDF5 format."""
    angmin: float = Field(..., description="Minimum voltage angle difference (radians)")
    angmax: float = Field(..., description="Maximum voltage angle difference (radians)")
    br_r: float = Field(..., description="Series resistance (p.u.)")
    br_x: float = Field(..., description="Series reactance (p.u.)")
    b_fr: float = Field(..., description="Half of br_b (p.u.)")
    b_to: float = Field(..., description="Half of br_b (p.u.)")
    rate_a: float = Field(..., description="Long-term thermal rating (MVA)")
    rate_b: float = Field(..., description="Short-term thermal rating (MVA)")
    rate_c: float = Field(..., description="Emergency thermal rating (MVA)")
    br_status: Optional[float] = Field(None, description="Branch status")
    tap: float = Field(..., description="Transformer tap ratio (p.u.)")
    shift: float = Field(..., description="Phase shift angle (radians)")

class H5BusSolution(OPFSchemaModel):
    """Bus solution features in HDF5 format."""
    va: float = Field(..., description="Voltage angle (radians)")
    vm: float = Field(..., description="Voltage magnitude (p.u.)")

class H5GeneratorSolution(OPFSchemaModel):
    """Generator solution features in HDF5 format."""
    pg: float = Field(..., description="Active power generation (p.u.)")
    qg: float = Field(..., description="Reactive power generation (p.u.)")

class H5EdgeSolution(OPFSchemaModel):
    """Edge solution (AC Line/Transformer) features in HDF5 format."""
    pf: float = Field(..., description="Active power flow (from → to) (p.u.)")
    qf: float = Field(..., description="Reactive power flow (from → to) (p.u.)")
    pt: float = Field(..., description="Active power flow (to → from) (p.u.)")
    qt: float = Field(..., description="Reactive power flow (to → from) (p.u.)")
