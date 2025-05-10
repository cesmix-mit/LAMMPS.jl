struct UnsafeNeighListVec <: AbstractVector{Cint}
    numneigh::Int
    neighbors::Ptr{Int32}
end

function Base.getindex(nle::UnsafeNeighListVec, i::Integer)
    @boundscheck checkbounds(nle, i)
    return unsafe_load(nle.neighbors, i)+Cint(1)
end

Base.size(nle::UnsafeNeighListVec) = (nle.numneigh,)

struct NeighList <: AbstractVector{Pair{Int32, UnsafeNeighListVec}}
    lmp::LMP
    idx::Cint
end

function Base.getindex(nl::NeighList, element::Integer)
    iatom = Ref{Cint}()
    numneigh = Ref{Cint}()
    neighbors = Ref{Ptr{Cint}}()
    @inline API.lammps_neighlist_element_neighbors(nl.lmp, nl.idx, element-one(element) #= 0-based indexing =#, iatom, numneigh, neighbors)
    @boundscheck iatom[] == -1 && throw(BoundsError(nl, element))
    return iatom[]+Cint(1) => UnsafeNeighListVec(numneigh[], neighbors[])
end

Base.size(nl::NeighList) = (API.lammps_neighlist_num_elements(nl.lmp, nl.idx),)

"""
    compute_neighborlist(lmp::LMP, id::String; request=0)

Retrieve neighbor list requested by a compute.

The neighbor list request from a compute is identified by the compute ID and the request ID.
The request ID is typically 0, but will be > 0 in case a compute has multiple neighbor list requests.

Each neighbor list contains vectors of local indices of neighboring atoms.
These can be used to index into Arrays returned from `extract_atom`.
"""
function compute_neighborlist(lmp::LMP, id::String; request=0)
    idx = API.lammps_find_compute_neighlist(lmp, id, request)
    idx == -1 && throw(KeyError(id))
    return NeighList(lmp, idx)
end

"""
    fix_neighborlist(lmp::LMP, id::String; request=0)

Retrieve neighbor list requested by a fix.

The neighbor list request from a fix is identified by the fix ID and the request ID.
The request ID is typically 0, but will be > 0 in case a fix has multiple neighbor list requests.

Each neighbor list contains vectors of local indices of neighboring atoms.
These can be used to index into Arrays returned from `extract_atom`.
"""
function fix_neighborlist(lmp::LMP, id::String; request=0)
    idx = API.lammps_find_compute_neighlist(lmp, id, request)
    idx == -1 && throw(KeyError(id))
    return NeighList(lmp, idx)
end

"""
    pair_neighborlist(lmp::LMP, style::String; exact=false, nsub=0, request=0)

This function determines which of the available neighbor lists for pair styles matches the given conditions. It first matches the style name. If exact is true the name must match exactly,
if exact is false, a regular expression or sub-string match is done. If the pair style is hybrid or hybrid/overlay the style is matched against the sub styles instead.
If a the same pair style is used multiple times as a sub-style, the nsub argument must be > 0 and represents the nth instance of the sub-style (same as for the pair_coeff command, for example).
In that case nsub=0 will not produce a match and this function will Error.

The final condition to be checked is the request ID (reqid). This will normally be 0, but some pair styles request multiple neighbor lists and set the request ID to a value > 0.

Each neighbor list contains vectors of local indices of neighboring atoms.
These can be used to index into Arrays returned from `extract_atom`.

## Examples
```julia
lmp = LMP()

command(lmp, \"""
    region cell block 0 3 0 3 0 3
    create_box 1 cell
    lattice sc 1
    create_atoms 1 region cell
    mass 1 1

    pair_style zero 1.0
    pair_coeff * *

    run 1
\""")

x = extract_atom(lmp, "x", LAMMPS_DOUBLE_2D; with_ghosts=true)

for (iatom, neighs) in pair_neighborlist(lmp, "zero")
    for jatom in neighs
        ix = @view x[:, iatom]
        jx = @view x[:, jatom]

        println(ix => jx)
   end
end
```
"""
function pair_neighborlist(lmp::LMP, style::String; exact=false, nsub=0, request=0)
    idx = API.lammps_find_pair_neighlist(lmp, style, exact, nsub, request)
    idx == -1 && throw(KeyError(style))
    return NeighList(lmp, idx)
end

struct UnsafeTopologyList{N} <: AbstractVector{NTuple{N, Cint}}
    ptr::Ptr{NTuple{N, Cint}}
    length::Int
end

Base.size(self::UnsafeTopologyList) = (self.length, )
function Base.getindex(self::UnsafeTopologyList, i)
    @boundscheck checkbounds(self, i)
    data = unsafe_load(self.ptr, i)
    atoms = Base.front(data) .+ Cint(1) # account for 0-based indexing
    return (atoms..., last(data))
end

function _topology_neighborlist(lmp, name, N)
    T = Ptr{NTuple{N, Cint}}
    length::Int = extract_setting(lmp, "n$(name)list")
    length == 0 && return UnsafeTopologyList(T(C_NULL), length)
    ptr::Ptr{T} = API.lammps_extract_global(lmp, "neigh_$(name)list")
    UnsafeTopologyList(unsafe_load(ptr), length)
end

bond_neighborlist(lmp::LMP) = @inline _topology_neighborlist(lmp, :bond, 3)
angle_neighborlist(lmp::LMP) = @inline _topology_neighborlist(lmp, :angle, 4)